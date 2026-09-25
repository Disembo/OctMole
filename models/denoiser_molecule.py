import torch
import torch.nn.functional as F
import warnings
from datasets.molecule import ATOM_TO_ID
from models.denoiser import Denoiser
from models.dit import DiT_models
from models.dit_molecule import MoleculeDiT_models, MoleculeOctreeDiT_models


MAX_OCTREE_NODES = 1000


class MoleculeVoxelDenoiser(Denoiser):
    def __init__(
        self,
        args
    ):
        super().__init__(args)
        assert args.train_mode == "molecule_voxel"
        self.data_dim = 3
        self.in_channels = 1
        self.spatial_size = args.voxel_size
        self.self_condition = bool(getattr(args, "self_condition", False))
        net_in_channels = self.in_channels * (2 if self.self_condition else 1)
        self.net = DiT_models[args.model](
            input_size=self.spatial_size,
            in_channels=net_in_channels,
            out_channels=self.in_channels,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            spatial_dims=self.data_dim,
        )
    def _net_forward(self, x, t):
        return self.net(x, t.flatten())

    def forward(self, x, global_step=None):
        x = x.float() * 2.0 - 1.0
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        if self.self_condition:
            if self.training and torch.rand((), device=x.device) < 0.5:
                with torch.no_grad():
                    x_self_cond = self._net_forward(
                        torch.cat([z, torch.zeros_like(z)], dim=1),
                        t
                    )
            else:
                x_self_cond = torch.zeros_like(z)
            x_pred = self._net_forward(
                torch.cat([z, x_self_cond.detach()], dim=1),
                t
            )
        else:
            x_pred = self._net_forward(z, t)

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        main_loss = loss.flatten(1).mean(dim=1).mean()
        loss_dict = self._training_metrics(x_pred, x, main_loss)
        return main_loss, loss_dict

    def _training_metrics(self, x_pred, x_target, main_loss):
        metrics = {}
        pred_prob = ((x_pred.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
        target_prob = ((x_target.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
        metrics.update(
            {
                "pred_pos_count": (pred_prob > 0.5).flatten(1).sum(dim=1).float().mean(),
                "pred_soft_pos_count": pred_prob.flatten(1).sum(dim=1).mean(),
                "target_pos_count": (target_prob > 0.5).flatten(1).sum(dim=1).float().mean(),
            }
        )
        return metrics

    @torch.no_grad()
    def generate(self, batch_size, device):
        sample_shape = (batch_size, self.in_channels) + (self.spatial_size,) * self.data_dim
        z = self.noise_scale * torch.randn(sample_shape, device=device)
        x_self_cond = torch.zeros_like(z)
        timestep_shape = (-1,) + (1,) * z.ndim
        expand_shape = (-1, batch_size) + (1,) * (z.ndim - 1)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(timestep_shape).expand(expand_shape)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z, x_self_cond = stepper(z, t, t_next, x_self_cond)
        # last step euler
        z, _ = self._euler_step(z, timesteps[-2], timesteps[-1], x_self_cond)
        return ((z + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def _forward_sample(self, z, t, x_self_cond=None):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(z)
            net_input = torch.cat([z, x_self_cond], dim=1)
        else:
            net_input = z
        # conditional
        x = self._net_forward(net_input, t)
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)
        x_pred = z + (1.0 - t).clamp_min(self.t_eps) * v
        return v, x_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, x_self_cond=None):
        v_pred, x_pred = self._forward_sample(z, t, x_self_cond)
        z_next = z + (t_next - t) * v_pred
        return z_next, x_pred.detach()

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, x_self_cond=None):
        v_pred_t, x_pred_t = self._forward_sample(z, t, x_self_cond)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next, _ = self._forward_sample(z_next_euler, t_next, x_pred_t.detach())

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next, x_pred_t.detach()


class MoleculeOctreeDenoiser(Denoiser):
    def __init__(self, args):
        super().__init__(args)
        assert args.train_mode == "molecule_octree"
        self.max_depth = int(args.octree_depth)
        self.depth_increment = int(getattr(args, "octree_depth_increment", 1))
        assert self.max_depth % self.depth_increment == 0
        self.out_channels = 8 ** self.depth_increment
        self.self_condition = bool(getattr(args, "self_condition", False))
        self.net = MoleculeOctreeDiT_models[args.model](
            max_depth=self.max_depth,
            depth_increment=self.depth_increment,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            rope_coord_scale=args.rope_coord_scale,
            self_condition=self.self_condition,
        )

    def forward(self, batch, global_step=None):
        target = batch["target"].float() * 2.0 - 1.0
        spvoxel = batch["spvoxel"]
        depth = batch["depth"]
        padding_mask = batch["padding_mask"]
        valid = ~padding_mask
        t = self.sample_t(target.shape[0], device=target.device)
        t_view = t[:, None, None]
        noise = torch.randn_like(target) * self.noise_scale
        z = t_view * target + (1.0 - t_view) * noise
        v = (target - z) / (1.0 - t_view).clamp_min(self.t_eps)
        self_cond = None
        if self.self_condition:
            self_cond = torch.zeros_like(target)
            if self.training and torch.rand((), device=target.device) < 0.5:
                with torch.no_grad():
                    self_cond = self.net(z, spvoxel, t, depth, padding_mask, self_cond)
        pred = self.net(z, spvoxel, t, depth, padding_mask, self_cond)
        v_pred = (pred - z) / (1.0 - t_view).clamp_min(self.t_eps)
        node_loss = ((v - v_pred) ** 2).mean(dim=-1)
        sample_loss = (node_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        loss = sample_loss.mean()
        return loss, self._metrics(pred, target, sample_loss, depth, t, valid)

    def _metrics(self, pred, target, sample_loss, depth, t, valid):
        pred_bit = pred.detach() > 0.0
        target_bit = target > 0.0
        bit_valid = valid.unsqueeze(-1)
        tp = (pred_bit & target_bit & bit_valid).sum().float()
        fp = (pred_bit & ~target_bit & bit_valid).sum().float()
        fn = (~pred_bit & target_bit & bit_valid).sum().float()
        metrics = {
            "bit_precision": tp / (tp + fp).clamp_min(1.0),
            "bit_recall": tp / (tp + fn).clamp_min(1.0),
            "bit_f1": 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0),
            "bit_iou": tp / (tp + fp + fn).clamp_min(1.0),
            "target_child_count": (target_bit & bit_valid).sum().float() / valid.sum().clamp_min(1),
            "pred_child_count": (pred_bit & bit_valid).sum().float() / valid.sum().clamp_min(1),
        }
        for d in range(self.depth_increment, self.max_depth + 1, self.depth_increment):
            mask = (depth == d).float()
            depth_valid = bit_valid & (depth == d)[:, None, None]
            depth_tp = (pred_bit & target_bit & depth_valid).sum().float()
            depth_fp = (pred_bit & ~target_bit & depth_valid).sum().float()
            depth_fn = (~pred_bit & target_bit & depth_valid).sum().float()
            metrics[f"bit_f1_depth_{d}"] = (
                2.0 * depth_tp
                / (2.0 * depth_tp + depth_fp + depth_fn).clamp_min(1.0)
            )
            metrics[f"loss_depth_{d}"] = (
                (sample_loss * mask).sum() / mask.sum().clamp_min(1.0)
            ).detach()
        for i, (lo, hi) in enumerate(zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.0))):
            mask = ((t >= lo) & (t < hi if i < 3 else t <= hi)).float()
            metrics[f"loss_t_{i}"] = (
                (sample_loss * mask).sum() / mask.sum().clamp_min(1.0)
            ).detach()
        return metrics

    @torch.no_grad()
    def _sample_level(self, spvoxel, padding_mask, depth):
        bsz, length, _ = spvoxel.shape
        z = self.noise_scale * torch.randn(
            bsz, length, self.out_channels, device=spvoxel.device
        )
        self_cond = torch.zeros_like(z) if self.self_condition else None
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=spvoxel.device)
        for i in range(self.steps - 1):
            t = timesteps[i].expand(bsz)
            t_next = timesteps[i + 1].expand(bsz)
            v, pred = self._forward_sample(z, spvoxel, t, depth, padding_mask, self_cond)
            z_euler = z + (t_next - t)[:, None, None] * v
            if self.method == "heun":
                v_next, _ = self._forward_sample(
                    z_euler, spvoxel, t_next, depth, padding_mask, pred.detach()
                )
                v = 0.5 * (v + v_next)
            elif self.method != "euler":
                raise NotImplementedError
            z = z + (t_next - t)[:, None, None] * v
            self_cond = pred.detach()
        t = timesteps[-2].expand(bsz)
        t_next = timesteps[-1].expand(bsz)
        v, _ = self._forward_sample(z, spvoxel, t, depth, padding_mask, self_cond)
        return z + (t_next - t)[:, None, None] * v

    def _forward_sample(self, z, spvoxel, t, depth, padding_mask, self_cond=None):
        pred = self.net(z, spvoxel, t, depth, padding_mask, self_cond)
        v = (pred - z) / (1.0 - t[:, None, None]).clamp_min(self.t_eps)
        return v, pred

    @staticmethod
    def _pad_nodes(nodes, device):
        max_len = max(max(len(x), 1) for x in nodes)
        spvoxel = torch.zeros(len(nodes), max_len, 3, dtype=torch.long, device=device)
        padding_mask = torch.ones(len(nodes), max_len, dtype=torch.bool, device=device)
        for i, value in enumerate(nodes):
            if len(value):
                spvoxel[i, :len(value)] = value
                padding_mask[i, :len(value)] = False
        return spvoxel, padding_mask

    def _record_macro_stats(
        self, parent, children, start_depth, end_depth, node_counts, branching
    ):
        previous = parent
        for d in range(start_depth + 1, end_depth + 1):
            width = 1 << (self.max_depth - d)
            current = torch.unique((children // width) * width, dim=0)
            counts = torch.zeros(len(previous), dtype=torch.long, device=children.device)
            if len(current):
                current_parents = (current // (width << 1)) * (width << 1)
                resolution = 1 << self.max_depth
                weights = torch.tensor(
                    [resolution * resolution, resolution, 1],
                    dtype=torch.long, device=children.device,
                )
                previous_keys = (previous * weights).sum(dim=1)
                parent_keys = (current_parents * weights).sum(dim=1)
                order = previous_keys.argsort()
                positions = torch.searchsorted(previous_keys[order], parent_keys)
                counts[order] = torch.bincount(positions, minlength=len(previous))
            branching[d - 1] = torch.bincount(counts, minlength=9)
            node_counts[d] = len(current)
            previous = current

    @torch.no_grad()
    def generate(self, batch_size, device, init_nodes=None, init_depth=0):
        side = 1 << self.depth_increment
        indices = torch.arange(self.out_channels, device=device)
        offsets = torch.stack(
            [indices % side, (indices // side) % side, indices // (side * side)],
            dim=1,
        )
        if init_nodes is not None:
            if init_depth % self.depth_increment:
                raise ValueError("init_depth must be a multiple of depth_increment")
            nodes = [n.to(device) for n in init_nodes]
            batch_size = len(nodes)
        else:
            nodes = [torch.zeros(1, 3, dtype=torch.long, device=device) for _ in range(batch_size)]
        node_counts = torch.zeros(batch_size, self.max_depth + 1, dtype=torch.long, device=device)
        branching = torch.zeros(batch_size, self.max_depth, 9, dtype=torch.long, device=device)
        node_counts[:, 0] = 1
        if init_nodes is not None and init_depth > 0:
            root = torch.zeros(1, 3, dtype=torch.long, device=device)
            for i, init in enumerate(nodes):
                self._record_macro_stats(
                    root, init, 0, init_depth, node_counts[i], branching[i],
                )
        cap_events = 0
        max_proposed = 0
        for d in range(self.depth_increment + init_depth, self.max_depth + 1, self.depth_increment):
            spvoxel, padding_mask = self._pad_nodes(nodes, device)
            depth = torch.full((batch_size,), d, dtype=torch.long, device=device)
            pred = self._sample_level(spvoxel, padding_mask, depth)
            child_width = 1 << (self.max_depth - d)
            next_nodes = []
            for i, parent in enumerate(nodes):
                scores = pred[i, :len(parent)].flatten()
                selected = (scores > 0.0).nonzero(as_tuple=True)[0]
                proposed = selected.numel()
                if proposed > MAX_OCTREE_NODES:
                    selected = torch.topk(scores, MAX_OCTREE_NODES).indices.sort().values
                    cap_events += 1
                    max_proposed = max(max_proposed, proposed)
                parent_index = selected // self.out_channels
                child_index = selected % self.out_channels
                children = parent[parent_index] + offsets[child_index] * child_width
                next_nodes.append(children)
                self._record_macro_stats(
                    parent, children, d - self.depth_increment, d,
                    node_counts[i], branching[i],
                )
            nodes = next_nodes
        if cap_events:
            warnings.warn(
                f"Octree generation capped {cap_events} sample-depth expansions at "
                f"{MAX_OCTREE_NODES} nodes; largest proposed node count was {max_proposed}."
            )
        spvoxel, padding_mask = self._pad_nodes(nodes, device)
        return {
            "spvoxel": spvoxel,
            "padding_mask": padding_mask,
            "node_counts": node_counts,
            "branching_hist": branching,
        }

class MoleculeFeatureDenoiser(Denoiser):
    def __init__(self, args):
        super().__init__(args)
        assert args.train_mode == "molecule_feature"
        self.atom_type_loss_weight = float(getattr(args, "atom_type_loss_weight", 1.0))
        self.coord_loss_weight = 1.0
        self.net = MoleculeDiT_models[args.model](
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            rope_coord_scale=(
                args.rope_coord_scale
                if getattr(args, "molecule_rope_scale_enabled", False)
                else 1.0
            ),
        )
        symbols = (
            ("H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi")
            if "drug" in str(args.data_path).lower()
            else ("H", "C", "N", "O", "F")
        )
        allowed_type_mask = torch.zeros(self.net.type_vocab_size, dtype=torch.bool)
        allowed_type_mask[[ATOM_TO_ID[symbol] for symbol in symbols]] = True
        self.register_buffer("allowed_type_mask", allowed_type_mask, persistent=False)

    def _uniform_types(self, shape):
        allowed = self.allowed_type_mask.nonzero(as_tuple=True)[0]
        return allowed[torch.randint(len(allowed), shape, device=allowed.device)]

    def _mask_type_logits(self, logits):
        return logits.masked_fill(~self.allowed_type_mask, float("-inf"))

    def forward(self, batch, global_step=None):
        atom_type = batch["atom_type"]
        coord = batch["coord_feature"]
        spvoxel = batch["spvoxel"]
        padding_mask = batch["padding_mask"]
        valid = ~padding_mask

        t = self.sample_t(coord.shape[0], device=coord.device).view(-1, 1, 1)
        noise = torch.randn_like(coord) * self.noise_scale
        z = t * coord + (1.0 - t) * noise
        v = (coord - z) / (1.0 - t).clamp_min(self.t_eps)

        type_noise = self._uniform_types(atom_type.shape)
        type_input = torch.where(
            torch.rand_like(atom_type, dtype=torch.float32) < t.squeeze(-1),
            atom_type,
            type_noise,
        )
        type_logits, coord_pred = self.net(
            type_input,
            z,
            spvoxel,
            t.flatten(),
            padding_mask=padding_mask,
        )
        type_logits = self._mask_type_logits(type_logits)
        v_pred = (coord_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        coord_loss = ((v - v_pred) ** 2).sum(dim=-1)
        coord_loss = (coord_loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        type_loss = F.cross_entropy(type_logits[valid], atom_type[valid])
        loss = self.coord_loss_weight * coord_loss + self.atom_type_loss_weight * type_loss
        with torch.no_grad():
            pred_type = type_logits.argmax(dim=-1)
            type_acc = ((pred_type == atom_type) & valid).float().sum() / valid.float().sum().clamp_min(1.0)
        return loss, {
            "coord_loss": coord_loss.detach(),
            "type_loss": type_loss.detach(),
            "type_acc": type_acc.detach(),
            "atom_count": valid.float().sum(dim=1).mean().detach(),
        }

    @torch.no_grad()
    def generate(self, batch):
        spvoxel = batch["spvoxel"]
        padding_mask = batch["padding_mask"]
        valid = ~padding_mask
        bsz, seqlen, _ = spvoxel.shape
        coord = self.noise_scale * torch.randn(bsz, seqlen, 4, device=spvoxel.device)
        atom_type = self._uniform_types((bsz, seqlen)).masked_fill(padding_mask, 0)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=spvoxel.device)
        
        for i in range(self.steps):
            t = timesteps[i].expand(bsz)
            t_next = timesteps[i + 1]
            type_logits, coord_pred = self.net(atom_type, coord, spvoxel, t, padding_mask=padding_mask)
            type_logits = self._mask_type_logits(type_logits)
            endpoint_probs = F.softmax(type_logits.float(), dim=-1)
            current = atom_type.unsqueeze(-1)
            step_probs = endpoint_probs * ((t_next - timesteps[i]) / (1.0 - timesteps[i]))
            step_probs.scatter_(-1, current, 0.0)
            step_probs.scatter_(-1, current, (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0))
            atom_type = torch.multinomial(
                step_probs.flatten(0, 1), 1
            ).view(bsz, seqlen).masked_fill(padding_mask, 0)

            v = (coord_pred - coord) / (1.0 - t.view(-1, 1, 1)).clamp_min(self.t_eps)
            coord = coord + (t_next - timesteps[i]) * v

        logp = F.log_softmax(type_logits.float(), dim=-1)
        score = (
            logp.gather(-1, atom_type.unsqueeze(-1)).squeeze(-1) * valid
        ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        
        return {
            "atom_type": atom_type,
            "coord_feature": coord,
            "spvoxel": spvoxel,
            "padding_mask": padding_mask,
            "score": score,
        }

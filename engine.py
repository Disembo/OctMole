import math
import sys
import os
from pathlib import Path

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched
import copy
from util.voxel_metrics import evaluate_voxel_folder
from datasets.molecule import MoleculeDataset, molecule_collate
from util.molecule_export import export_molecule_sample, export_stage1_voxel_ply, write_xyz_from_sample_dirs
from util.molecule_metrics import evaluate_xyz_file


def _dist_barrier():
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def _molecule_voxel_forward(batch, model, device, global_step):
    x = batch.to(device, non_blocking=True).to(torch.float32)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, global_step=global_step)

    if isinstance(loss, tuple):  # loss tensor and loss dict
        return loss
    return loss, {}


def _molecule_feature_forward(batch, model, device, global_step):
    batch = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch, global_step=global_step)
    if isinstance(loss, tuple):
        return loss
    return loss, {}


def _molecule_octree_forward(batch, model, device, global_step):
    return _molecule_feature_forward(batch, model, device, global_step)


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        global_step = epoch * len(data_loader) + data_iter_step
        
        if args.train_mode == "molecule_voxel":
            loss, loss_dict = _molecule_voxel_forward(batch, model, device, global_step)
        elif args.train_mode == "molecule_octree":
            loss, loss_dict = _molecule_octree_forward(batch, model, device, global_step)
        elif args.train_mode == "molecule_feature":
            loss, loss_dict = _molecule_feature_forward(batch, model, device, global_step)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        for name, value in loss_dict.items():
            metric_logger.update(**{name: float(value.item())})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_dict_reduce = {
            name: misc.all_reduce_mean(float(value.item()))
            for name, value in loss_dict.items()
        }

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
                for name, value in loss_dict_reduce.items():
                    log_writer.add_scalar('train/{}'.format(name), value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def _evaluate_molecule_voxel(model, args, save_folder, epoch, batch_size, log_writer):
    device = next(model.parameters()).device
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = math.ceil(args.num_samples / (batch_size * world_size))
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sampled = model.generate(batch_size, device)
        _dist_barrier()

        sampled = sampled.detach().cpu()

        # distributed save samples
        for b_id in range(sampled.size(0)):
            sample_id = i * sampled.size(0) * world_size + local_rank * sampled.size(0) + b_id
            if sample_id >= args.num_samples:
                break
            voxel = np.clip(sampled[b_id].numpy(), 0, 1).astype(np.float32)
            np.save(os.path.join(save_folder, '{}.npy'.format(str(sample_id).zfill(5))), voxel)
            export_stage1_voxel_ply(
                voxel,
                os.path.join(save_folder, '{}.ply'.format(str(sample_id).zfill(5))),
                threshold=getattr(args, "eval_voxel_threshold", 0.5),
            )

    _dist_barrier()
    if misc.get_rank() == 0:
        metrics = evaluate_voxel_folder(
            save_folder,
            train_index=os.path.join(args.data_path, "index", "occupancy_index_train.npz"),
            test_index=os.path.join(args.data_path, "index", "occupancy_index_valid.npz"),
            threshold=getattr(args, "eval_voxel_threshold", 0.5),
        )
        for name, value in metrics.items():
            print("{}: {:.6f}".format(name, value))
            if log_writer is not None:
                log_writer.add_scalar("eval/{}".format(name), value, epoch)


def _evaluate_molecule_feature(model, args, save_folder, epoch, batch_size, log_writer):
    device = next(model.parameters()).device
    dataset = MoleculeDataset(
        args.data_path,
        train_mode="molecule_feature",
        voxel_size=args.voxel_size,
        split="valid",
        box_extent=args.box_extent,
    )
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        len(dataset),
        size=args.num_samples,
        replace=args.num_samples > len(dataset),
    )
    local_ids = list(range(local_rank, args.num_samples, world_size))
    for offset in range(0, len(local_ids), batch_size):
        output_ids = local_ids[offset : offset + batch_size]
        batch = molecule_collate([dataset[i] for i in sample_indices[output_ids]])
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        result = model.generate(batch)
        result = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in result.items()}
        for row, sample_id in enumerate(output_ids):
            box_extent = float(batch["box_extent"][row].detach().cpu())
            export_molecule_sample(
                save_folder,
                {k: v[row] for k, v in result.items()} | {
                    "coord_feature_gt": batch["coord_feature"][row].detach().cpu().numpy()
                },
                sample_id,
                voxel_size=args.voxel_size,
                box_extent=box_extent,
            )

    _dist_barrier()
    if misc.get_rank() == 0:
        xyz_path, count = write_xyz_from_sample_dirs(save_folder, os.path.join(save_folder, "samples.xyz"))
        dataset_name = "drug" if "drug" in str(args.data_path).lower() else "qm9"
        metrics = evaluate_xyz_file(xyz_path, dataset=dataset_name)
        for name, value in metrics.items():
            print("{}: {:.6f}".format(name, value))
            if log_writer is not None:
                log_writer.add_scalar("eval/{}".format(name), value, epoch)


def _evaluate_molecule_octree(model, args, save_folder, epoch, batch_size, log_writer):
    device = next(model.parameters()).device
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = math.ceil(args.num_samples / (batch_size * world_size))
    for i in range(num_steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model.generate(batch_size, device)
        for row in range(batch_size):
            sample_id = i * batch_size * world_size + local_rank * batch_size + row
            if sample_id >= args.num_samples:
                break
            valid = ~result["padding_mask"][row]
            payload = {
                "spvoxel": result["spvoxel"][row, valid].cpu().numpy(),
                "node_counts": result["node_counts"][row].cpu().numpy(),
                "branching_hist": result["branching_hist"][row].cpu().numpy(),
            }
            np.savez_compressed(os.path.join(save_folder, f"{sample_id:05d}.npz"), **payload)
        _dist_barrier()
    if misc.get_rank() == 0 and log_writer is not None:
        rows = [np.load(path)["node_counts"] for path in sorted(Path(save_folder).glob("*.npz"))]
        depth_counts = np.stack(rows)
        log_writer.add_scalar("eval/leaf_count", depth_counts[:, -1].mean(), epoch)
        for d, value in enumerate(depth_counts.mean(axis=0)):
            log_writer.add_scalar(f"eval/node_count_depth_{d}", value, epoch)


@torch.no_grad()
def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()

    # Construct the folder name for saving generated samples.
    save_folder = os.path.join(
        args.output_dir,
        f"{model_without_ddp.method}-steps{model_without_ddp.steps}-{args.num_samples}"
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    _dist_barrier()

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    if args.train_mode == "molecule_voxel":
        _evaluate_molecule_voxel(model_without_ddp, args, save_folder, epoch, batch_size, log_writer)
    elif args.train_mode == "molecule_octree":
        _evaluate_molecule_octree(model_without_ddp, args, save_folder, epoch, batch_size, log_writer)
    elif args.train_mode == "molecule_feature":
        _evaluate_molecule_feature(model_without_ddp, args, save_folder, epoch, batch_size, log_writer)

    _dist_barrier()
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    _dist_barrier()

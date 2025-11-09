"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

Multi-GPU version with distributed computing support.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--max_samples', type=int, default=9999999999,
                    help='max data samples')
parser.add_argument('--num-workers', type=int, default=4,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('--world-size', type=int, default=-1,
                    help='Number of GPUs to use for distributed evaluation')
parser.add_argument('--rank', type=int, default=0,
                    help='Rank of the current process')
parser.add_argument('--dist-url', type=str, default='tcp://localhost:12355',
                    help='URL used to set up distributed training')
parser.add_argument('--dist-backend', type=str, default='nccl',
                    help='Distributed backend')
parser.add_argument('--local_rank', type=int, default=0,
                    help='Local rank for distributed training')
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class DistributedImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def setup(rank, world_size, dist_url='tcp://localhost:12355', backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = dist_url.split(':')[-1]
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def get_distributed_activations(files, model, batch_size=50, dims=2048, device='cpu',
                               num_workers=1, rank=0, world_size=1):
    """Calculates the activations of the pool_3 layer for all images using distributed computing."""
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # Split files across GPUs
    files_per_gpu = len(files) // world_size
    start_idx = rank * files_per_gpu
    end_idx = start_idx + files_per_gpu if rank < world_size - 1 else len(files)
    local_files = files[start_idx:end_idx]
    
    if len(local_files) == 0:
        return np.empty((0, dims))

    trans = TF.Compose([
        TF.Resize(299),
        TF.CenterCrop(299),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    dataset = DistributedImagePathDataset(local_files, transforms=trans)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers,
                                             pin_memory=True)

    pred_arr = np.empty((len(local_files), dims))
    start_idx = 0

    for batch in tqdm(dataloader, desc=f'Rank {rank}'):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + 
            np.trace(sigma2) - 2 * tr_covmean)


def compute_distributed_statistics_of_path(path, model, batch_size, dims, device, 
                                          max_samples, num_workers, rank, world_size):
    """Compute statistics using distributed computing."""
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
        return m, s
    
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('*.{}'.format(ext))])
    
    if len(files) > max_samples:
        print(f'Restrict to {len(files)} to {max_samples} images')
        files = files[:max_samples]
    
    print(f'{path} has {len(files)} images, rank {rank} processing {len(files) // world_size} images')
    
    # Get activations from all GPUs
    local_activations = get_distributed_activations(files, model, batch_size, dims, 
                                                   device, num_workers, rank, world_size)
    
    # Gather all activations from all processes
    if world_size > 1:
        # Convert to tensor for distributed gathering
        local_activations_tensor = torch.from_numpy(local_activations).to(device)
        
        # Gather sizes first to create proper tensor
        local_size = torch.tensor([local_activations_tensor.size(0)], device=device)
        sizes = [torch.tensor([0], device=device) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        
        max_size = max(sizes).item()
        # Pad tensors to max size
        if local_activations_tensor.size(0) < max_size:
            padding = torch.zeros(max_size - local_activations_tensor.size(0), 
                                local_activations_tensor.size(1), device=device)
            local_activations_tensor = torch.cat([local_activations_tensor, padding], dim=0)
        
        # Gather all padded tensors
        gathered_tensors = [torch.zeros_like(local_activations_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, local_activations_tensor)
        
        # Combine and remove padding
        all_activations = []
        for i, tensor in enumerate(gathered_tensors):
            actual_size = sizes[i].item()
            all_activations.append(tensor[:actual_size].cpu().numpy())
        
        all_activations = np.concatenate(all_activations, axis=0)
    else:
        all_activations = local_activations
    
    # Only rank 0 computes final statistics
    if rank == 0:
        mu = np.mean(all_activations, axis=0)
        sigma = np.cov(all_activations, rowvar=False)
        return mu, sigma
    else:
        return None, None


def calculate_fid_given_paths_distributed(rank, world_size, args):
    """Distributed FID calculation."""
    setup(rank, world_size, args.dist_url, args.dist_backend)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Compute statistics for both paths
    m1, s1 = compute_distributed_statistics_of_path(args.path[0], model, args.batch_size,
                                                   args.dims, device, args.max_samples,
                                                   args.num_workers, rank, world_size)
    
    m2, s2 = compute_distributed_statistics_of_path(args.path[1], model, args.batch_size,
                                                   args.dims, device, args.max_samples,
                                                   args.num_workers, rank, world_size)
    
    # Only rank 0 computes final FID
    if rank == 0:
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('FID: ', fid_value)
        
        # Save result to file
        result_file = os.path.join(os.path.dirname(args.path[0]), 'fid_result.json')
        with open(result_file, 'w') as f:
            json.dump({'fid': float(fid_value)}, f, indent=2)
    
    cleanup()


def main():
    args = parser.parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
    
    if args.world_size > 1:
        print(f'Using {args.world_size} GPUs for distributed FID evaluation')
        mp.spawn(calculate_fid_given_paths_distributed,
                args=(args.world_size, args),
                nprocs=args.world_size,
                join=True)
    else:
        # Single GPU fallback
        if args.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        if args.num_workers is None:
            try:
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count()
            num_workers = min(num_cpus, 8) if num_cpus is not None else 0
        else:
            num_workers = args.num_workers
        
        if args.save_stats:
            # Single GPU save stats (modify as needed for distributed)
            pass
        else:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
            model = InceptionV3([block_idx]).to(device)
            
            m1, s1 = compute_distributed_statistics_of_path(args.path[0], model, args.batch_size,
                                                           args.dims, device, args.max_samples,
                                                           num_workers, 0, 1)
            m2, s2 = compute_distributed_statistics_of_path(args.path[1], model, args.batch_size,
                                                           args.dims, device, args.max_samples,
                                                           num_workers, 0, 1)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
            print('FID: ', fid_value)


if __name__ == '__main__':
    main()
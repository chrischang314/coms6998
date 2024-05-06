import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image

# packages for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cudnn.enabled = False
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import random

# Hyperparameters
k = 5  # Number of previous epochs to use for stopping criterion
alpha = 10  # Threshold for stopping criterion
s_min = 4  # Minimum batch size
s_max = 128  # Maximum batch size
target_train_time = 60  # Target training time per epoch in seconds

class RandomResizeTransform:
    def __init__(self, min_scale=0.5, max_scale=1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = int(32 * scale)
        transform = transforms.Compose([
            transforms.Resize(new_size),  # Resize image to new_size
            transforms.Pad((32 - new_size) // 2) if new_size < 32 else transforms.CenterCrop(32)  # Pad or crop to maintain 32x32
        ])
        return transform(img)

# Load CIFAR-10 dataset with random resizing transform
#transform = transforms.Compose([
#    RandomResizeTransform(),
#    transforms.ToTensor()
#])
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # Correctly convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for pre-trained models
])


dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def stopping_criterion(train_losses):
    """
    Compute the stopping criterion P_k(t) based on the past k train losses
    """
    total_loss = sum(train_losses)
    min_loss = min(train_losses)
    p_k = 1000 * ((total_loss / min_loss) - 1)
    return p_k

def adjust_batch_size(curr_train_loss, prev_train_losses, train_time, world_size, batch_size):
    prev_train_losses.append(curr_train_loss)
    if len(prev_train_losses) > k:
        prev_train_losses.pop(0)

    p_k = stopping_criterion(prev_train_losses)

    # Gather training time from all processes
    train_times = [torch.tensor([0.0]).cuda() for _ in range(world_size)]
    torch.distributed.all_gather(train_times, torch.tensor([train_time]).cuda())
    avg_train_time = sum(train_times) / world_size

    if p_k < alpha or avg_train_time > target_train_time:
        batch_size *= 2
    else:
        batch_size /= 2

    return batch_size

def train_model(rank, args):
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, args.world_size)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # instantiate the model and transfer it to the GPU
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10's 10 classes

    model = model.to(rank)
    # wraps the network around distributed package
    model = DDP(model, device_ids=[rank])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Preparing the training data
    transforms_train = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset,
                                                                          num_replicas=args.world_size, rank=rank)

    trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0, pin_memory=True,
                                              sampler=train_data_sampler)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("training")
    target_loss = 0.5  # Adjust this value based on your requirements

    # Training
    batch_size = args.batch_size
    prev_train_losses = []  # Store previous k train losses
    for epoch in range(args.n_epochs):
        start_epoch_time = time.time()

        train_data_sampler.set_epoch(epoch)
        trainloader.batch_sampler.sampler = train_data_sampler

        model.train()
        train_loss = 0
        accuracy = 0
        total = 0

        start_train_time = time.time()
        for images, labels in trainloader:
            # Move data to the appropriate device
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for param in model.parameters():
              param.grad.data.div_(trainloader.batch_size)
            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            _, prediction = outputs.max(1)
            accuracy += prediction.eq(labels).sum().item()
        end_train_time = time.time()

        train_time = end_train_time - start_train_time

        # Gather train loss from all processes
        losses = [torch.tensor([0.0]).cuda(rank) for _ in range(args.world_size)]
        torch.distributed.all_gather(losses, torch.tensor([train_loss]).cuda(rank))
        avg_loss = sum(losses) / args.world_size

        # Adjust batch size based on the stopping criterion and training time
        new_batch_size = adjust_batch_size(avg_loss.item(), prev_train_losses, train_time, args.world_size, trainloader.batch_size)

        # Limit the batch size to a reasonable range
        new_batch_size = max(new_batch_size, 4)  # Minimum batch size
        new_batch_size = min(new_batch_size, 128)  # Maximum batch size

        if new_batch_size != trainloader.batch_size:
            # Re-create the data loaders with the new batch size
            train_data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=dataset, num_replicas=args.world_size, rank=rank)
            trainloader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=new_batch_size, shuffle=False,
                num_workers=0, pin_memory=True, sampler=train_data_sampler)

        if rank == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Training Accuracy: {accuracy / total:.4f}, "
                  f"Time: {end_train_time - start_train_time:.2f}s")

        end_epoch_time = time.time()

        print(f"Total Epoch Time: {end_epoch_time - start_epoch_time:.2f}, {trainloader.batch_size}, rank: {rank}")

    print("Training DONE!!!")
    print()
    print('Testing BEGINS!!')


    cleanup()


def run_train_model(train_func, world_size):

    parser = argparse.ArgumentParser("PyTorch - Training ResNet101 on CIFAR10 Dataset")
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    parser.add_argument('--lr', default=0.01, type=float, help='Default Learning Rate')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--n_epochs', type=int, default=10, help='Total number of epochs for training')
    args = parser.parse_args()
    print(args)

    # this is responsible for spawning 'nprocs' number of processes of the train_func function with the given
    # arguments as 'args'
    mp.spawn(train_func, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    # since this example shows a single process per GPU, the number of processes is simply replaced with the
    # number of GPUs available for training.
    n_gpus = torch.cuda.device_count()
    run_train_model(train_model, n_gpus)

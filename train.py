import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchsummary  import summary
from torchvision.models import ViT_B_16_Weights
from custom_vision_transformer import PatchEmbed, Attention, MLP, Transformer, VisionTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Pytorch implementation of Vision Transformer fine-tuning.')
parser.add_argument('--data_dir', metavar='DIR', type=str, help='Path to training dataset.')
parser.add_argument('--dataset_name', metavar='NAME', type=str, help='Name of the dataset.')
parser.add_argument('--batch_sz', metavar='N', type=int, default=512, help='Size of each trainingg batch.')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer of the training process.')
parser.add_argument('--lr', metavar='N', type=float, default=1e-3, help='Learning rate of the optimizer.')
parser.add_argument('--momentum', metavar='N', type=float, default=0.9, help='Momentum of the SGD optimizer.')
parser.add_argument('--epochs', metavar='N', type=int, default=10, help='Number of epochs.')
parser.add_argument('--device', type=str, default='cuda', help='Device (accelerator) to use.')
parser.add_argument('--verbose', action='store_true', help='Display training process.')
parser.add_argument('--num_threads', metavar='N', type=int, default=0, help='Number of threads to be used to process parallel.')
parser.add_argument('--pretrained', action='store_true', help='Use pre-trained model.')
parser.add_argument('--model_dir', metavar='DIR', type=str, default=None, help='Path to pre-trained model.')
parser.add_argument('--checkpoint_dir', metavar='DIR', type=str, default=None, help='Path to pre-trained checkpoint.')
parser.add_argument('--keep_head', action='store_true', help='Stop replacing classification head with the new one.')
parser.add_argument('--train_all', action='store_true', help='Train all layers in the (pre-trained) model.')
parser.add_argument('--save_dir', metavar='DIR', type=str, help='Path to save model weight.')

parser.add_argument('--img_sz', metavar='N', type=int, default=384, help='Size of the image.')
parser.add_argument('--patch_sz', metavar='N', type=int, default=16, help='Size of each patch.')
parser.add_argument('--in_chans', metavar='N', type=int, default=3, help='Number of channels of each image.')
parser.add_argument('--emb_dim', metavar='N', type=int, default=768, help='Dimensionality of each token/patch embedding.')
parser.add_argument('--depth', metavar='N', type=int, default=12, help='Number of Transformer encoder blocks.')
parser.add_argument('--num_head', metavar='N', type=int, default=12, help='Number of self-attention heads.')
parser.add_argument('--mlp_ratio', metavar='N', type=float, default=4.0, help='Ratio of MLP hidden dimention to embedding dimention.')
parser.add_argument('--qkv_bias', action='store_true', help='Add bias to qkv projections.')
parser.add_argument('--qk_norm', action='store_true', help='Normalize the query and key.')
parser.add_argument('--attn_p', type=float, default=0.0, help='Probability of dropout block after attention mechanism.')
parser.add_argument('--proj_p', type=float, default=0.0, help='Probability of dropout block after linear projection.')

"""
Example:
>>> !python train.py --data_dir ./dataset --dataset_name food-101 --pretrained --model_dir ./Model/vit_base_patch16_384_custom.pth --epochs 5 --batch_sz 512 --device cuda:0 --verbose --save_dir ./Fine_tuning
"""

def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    save_dir: str,
    model_name: str=None,
    optimizer: str="sgd",
    lr: float=1e-3,
    momentum: float=0.9,
    epochs: int=10,
    device: str="cuda",
    verbose: bool=True,
) -> None:
    """
    Fine-tuning pre-trained model.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.
    
    train_dataloader : torch.utils.data.dataloader.DataLoader
        Dataloader for training.

    save_dir : str
        Path to save model weight.

    model_name : str, default=None
        Name of saved model if it exits.
        
    optimizer : str, default="sgd"
        Optimizer of the training process.
        Currently supported: "sgd" and "adam".

    lr : float, default=1e-3
        Fine-tuning learning rate of the optimizer.

    momentum : float, default=0.9
        Fine-tuning momentum of the SGD optimizer.

    epochs : int, default=10
        Number of epochs.

    device : str, default="cuda"
        Device (accelerator) to use for training.
        Currently supported: "cuda" and "cpu".

    verbose : bool, default=True
        Show training process.

    save_dir : str
        Path to save model weight.
    """

    if "cuda" in device :
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            print("GPU is not supported. CPU will be used instead.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")

    # Set device for model
    model.to(device)

    # Loss function for classification task
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")


    print("Start training..")
    model.train()

    it = range(epochs)
    ite = tqdm(it) if (verbose is None) else it

    for epoch in ite:
        start = time.time()
        train_loss = 0.0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Pass forward
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backpropagation and optimization
            # Reset gradients of all weights of the model to 0 before backpropagation
            optimizer.zero_grad()
            # Calculate gradient of the loss function of the corresponding weights
            loss.backward()
            # Updating weights in the optimization process
            optimizer.step()
            # Updating loss value
            train_loss = train_loss + loss.item()
        
        # Show training loss of each epoch
        average_train_loss = train_loss / len(train_dataloader)
        average_time = (time.time() - start) / len(train_dataloader)
        seconds = int(average_time)
        ms_seconds = int((average_time - seconds)*1e3)

        if verbose:
            print(f"Epoch {epoch}/{epochs - 1}:\t[===================================]\t- {seconds}s {ms_seconds}ms/it\t- Loss: {average_train_loss:.3f}")

        torch.save(model.state_dict(), f"{save_dir}/last.pth")

    if model_name is None:
        torch.save(model.state_dict(), f"{save_dir}/custom_model.pth")
    else:
        torch.save(model.state_dict(), f"{save_dir}/{model_name}_model.pth")

def main() -> None:
    args = parser.parse_args()
    
    if args.verbose:
        print("Load dataset.")
    pretrained_weight = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_transforms = pretrained_weight.transforms()
    
    train_dataset = None
    num_classes = 1000
    if args.dataset_name == "food-101":
        train_dataset = datasets.Food101(root=args.data_dir, split="train", transform=vit_transforms, download=True)
        num_classes = 101
    elif args.dataset_name == "flowers-102":
        train_dataset = datasets.Flowers102(root=args.data_dir, split="train", transform=vit_transforms, download=True)
        num_classes = 102
    elif args.dataset_name == "pcam-2":
        train_dataset = datasets.PCAM(root=args.data_dir, split="train", transform=vit_transforms, download=True)
        num_classes = 2
    elif args.dataset_name == "country-211":
        train_dataset = datasets.Country211(root=args.data_dir, split="train", transform=vit_transforms, download=True)
        num_classes = 211
    elif args.dataset_name == "oxfordiiipet-37":
        train_dataset = datasets.OxfordIIITPet(root=args.data_dir, split="trainval", transform=vit_transforms, download=True)
        num_classes = 37

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=args.num_threads)

    vit = None
    vit_custom_config = {
        "img_size": args.img_sz,
        "patch_size": args.patch_sz,
        "in_chans": args.in_chans,
        "num_classes": num_classes,
        "embed_dim": args.emb_dim,
        "depth": args.depth,
        "num_heads": args.num_head,
        "mlp_ratio": args.mlp_ratio,
        "qkv_bias": args.qkv_bias,
        "qk_norm": args.qk_norm,
        "attn_p": args.attn_p,
        "proj_p": args.proj_p,
    }
    if args.pretrained:
        if args.verbose:
            print("Load pre-trained model.")

        assert (args.model_dir is not None) or (args.checkpoint_dir is not None)
        if args.model_dir is not None:
            pretrained_vit = torch.load(args.model_dir)
        else:
            pretrained_vit = VisionTransformer(**vit_custom_config)
            pretrained_vit.load_checkpoint(args.checkpoint_dir)

        # Freeze pre-trained model
        if not args.train_all:
            for name, param in pretrained_vit.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
            # for param in pretrained_vit.parameters():
            #     param.requires_grad = False
        
        if not args.keep_head:
            embed_dim = args.emb_dim
            pretrained_vit.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            pretrained_vit.head = nn.Linear(embed_dim, num_classes)
        if args.verbose:
            summary(
                model=pretrained_vit, 
                input_data=torch.randn(1, args.in_chans, args.img_sz, args.img_sz),
                col_names=["input_size", "output_size", "num_params"]
            )
        vit = pretrained_vit
    else:
        if args.verbose:
            print("Config model.")
        vit = VisionTransformer(**vit_custom_config)

    train(
        model=vit,
        train_dataloader=train_dataloader,
        save_dir=args.save_dir,
        model_name=args.dataset_name,
        optimizer=args.opt,
        lr=args.lr,
        momentum=args.momentum,
        epochs=args.epochs,
        device=args.device,
        verbose=args.verbose
    )
    
if __name__ == '__main__':
    main()
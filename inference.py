import argparse
from Utils.preprocessing import Dataloader
import torch
from torch.nn.functional import softmax
from torchvision import datasets
from custom_vision_transformer import VisionTransformer
from torchvision.models import ViT_B_16_Weights

parser = argparse.ArgumentParser(description='Pytorch implementation of Vision Transformer.')
parser.add_argument('--img', metavar='DIR', type=str, default=None, help='Path to input image.')
parser.add_argument('--img_id', metavar='N', type=int, default=None, help='Id of the input image in the dataset in case there is no specific path to the image.')
parser.add_argument('--img_sz', metavar='N', type=int, default=384, help='Size of the image.')
parser.add_argument('--data_dir', metavar='DIR', type=str, default=None, help='Path to training dataset.')
parser.add_argument('--dataset_name', metavar='NAME', type=str, default=None, help='Name of the dataset.')
parser.add_argument('--dataset_type', metavar='NAME', type=str, default=None, help='Type of the dataset.')
parser.add_argument('--cls_dir', metavar='DIR', type=str, default=None, help='Path to file containing classes list.')
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
parser.add_argument('--pretrained', action='store_true', help='Use pre-trained model.')
parser.add_argument('--pretrained_model_dir', metavar='DIR', type=str, help='Path to pre-trained model.')
parser.add_argument('--checkpoint_dir', metavar='DIR', type=str, help='Path to pre-trained model checkpoint.')
parser.add_argument('--device', type=str, default='cuda', help='Device (accelerator) to use.')
parser.add_argument('--verbose', action='store_true', help='Display whole process.')

"""
Examples:
>>> str(test_dataset._image_files[idx])
# Output: 'dataset/food-101/images/churros/1148015.jpg'

# 1. Use directly classes list in the pytorch dataset and load pre-trained/fine-tuned model
>>> !python inference.py --img ./dataset/food-101/images/churros/1148015.jpg --data_dir ./dataset --dataset_name food101 --pretrained --pretrained_model_dir ./Model/vit_base_patch16_384_custom.pth --device cuda:0
# 2. Use via classes.txt and load pre-trained/fine-tuned checkpoint
>>> !python inference.py --img ./dataset/food-101/images/churros/1148015.jpg --cls_dir ./dataset/food-101/meta/classes.txt --pretrained --checkpoint_dir ./fine_tuning/model.pth --device cuda:0
"""

def main():
    args = parser.parse_args()

    pretrained_weight = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_transforms = pretrained_weight.transforms()
    
    test_dataset = None
    if (args.cls_dir is None) and (args.data_dir is not None):
        if args.dataset_name == "food-101":
            test_dataset = datasets.Food101(root=args.data_dir, split="test", transform=vit_transforms, download=True)
        elif args.dataset_name == "flowers-102":
            test_dataset = datasets.Flowers102(root=args.data_dir, split="test", transform=vit_transforms, download=True)
        elif args.dataset_name == "pcam-2":
            test_dataset = datasets.PCAM(root=args.data_dir, split="test", transform=vit_transforms, download=True)
        elif args.dataset_name == "country-211":
            test_dataset = datasets.Country211(root=args.data_dir, split="test", transform=vit_transforms, download=True)
        elif args.dataset_name == "oxfordiiipet-37":
            test_dataset = datasets.OxfordIIITPet(root=args.data_dir, split="test", transform=vit_transforms, download=True)

    loader = None
    if args.dataset_type != 'h5':
        loader = Dataloader(args.img, test_dataset, args.dataset_type, args.cls_dir, vit_transforms)
    else:
        data_input = f"{args.img}-{args.img_id}"
        loader = Dataloader(data_input, test_dataset, args.dataset_type, args.cls_dir, vit_transforms)
    img, img_tensor, classes = next(loader.data)

    if classes is None:
        num_cls = int(args.dataset_name.split("-")[1])
    else:
        num_cls = len(classes)

    #  Vision transformer Configuration
    vit_custom_config = {
        "img_size": args.img_sz,
        "patch_size": args.patch_sz,
        "in_chans": args.in_chans,
        "num_classes": num_cls,
        "embed_dim": args.emb_dim,
        "depth": args.depth,
        "num_heads": args.num_head,
        "mlp_ratio": args.mlp_ratio,
        "qkv_bias": args.qkv_bias,
        "qk_norm": args.qk_norm,
        "attn_p": args.attn_p,
        "proj_p": args.proj_p,
    }

    if args.verbose:
        print("Load model.")
    ViT = None
    # Load pre-trained model
    if args.pretrained:
        if args.pretrained_model_dir is not None:
            ViT = VisionTransformer().load_model(args.pretrained_model_dir)
        else:
            ViT = VisionTransformer(**vit_custom_config)
            ViT.load_checkpoint(args.checkpoint_dir)
    else:
        ViT = VisionTransformer(**vit_custom_config)

    device = None
    if "cuda" in args.device :
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            print("GPU is not supported. CPU will be used instead.")
            device = torch.device("cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")

    ViT.to(device=device)
    ViT.eval()

    if args.verbose:
        print("Start inferencing.")
    logits = ViT(img_tensor.to(device=device))
    probs = softmax(logits, dim=-1)

    top_probs, top_ixs = probs[0].topk(1)
    prob, idx = top_probs.item(), top_ixs.item()

    cls = idx
    if classes is not None:
        cls = classes[idx].strip()

    print(f"Prediction: {cls} ({(prob*100):.2f}%)")

if __name__ == '__main__':
    main()
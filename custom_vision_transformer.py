from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

class PatchEmbed(nn.Module):
    """
    Split 2D image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image.

    patch_size : int, default=16
        Size of a patch.

    in_chans : int, default=3
        Number of input channels.

    embed_dim : int, default=768
        The embedding dimension.

    Attributes
    ----------
    num_patches : int
        Number of patches inside the image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding.
    """

    def __init__(
        self, 
        img_size: Optional[int] = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels = in_chans, 
            out_channels = embed_dim, 
            kernel_size = patch_size, 
            stride = patch_size
        ) # Using the convolution with kernel size of 16x16 and stride=16 to split the image into patches of 16x16

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(num_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(num_samples, num_patches, embed_dim)`.
        """

        x = self.proj(x) # (num_samples, embed_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2) # (num_samples, embed_dim, num_patches)
        x = x.transpose(1, 2) # (num_samples, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    """
    Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and output dimension of each feature token.

    num_heads : int, default=12
        Number of attention heads.

    qkv_bias : bool, default=True
        If True then bias is included to the query, key, and value projections.

    qk_norm : bool, default=False
        If Tre then normalization is applied the query and key.

    attn_p : float, default=0.0
        Dropout probability applied to the query, key, and value tensors.

    proj_p : float, default=0.0
        Dropout probability applied to the output tensor.
        
    Attributes
    ----------
    scale : float
        Normalization constant for the dot product.
    
    qkv: nn.Linear
        Linear projection for the query, key, and value.

    proj : nn. Linear
        Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
        
    norm_layer : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_p: float = 0.0,
        proj_p: float = 0.0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.norm_layer = nn.LayerNorm
        self.q_norm = self.norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = self.norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.
        """

        num_samples, num_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (num_samples, n_patches + 1, 3*dim)
        qkv = qkv.reshape(
            num_samples, 
            num_tokens, 
            3, 
            self.num_heads, 
            self.head_dim
        ) # (num_samples,  n_patches + 1, 3, num_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, num_samples, num_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2] # (num_samples, num_heads, n_patches + 1, head_dim)
        q, k = self.q_norm(q), self.q_norm(k)
        k_T = k.transpose(-2, -1) # (num_samples, num_heads, head_dim, n_patches + 1)
        scaled_dot_product = (q @ k_T) * self.scale # (num_samples, num_heads, n_patches + 1, n_patches + 1)
        attn_weight = scaled_dot_product.softmax(dim=-1) # (num_samples, num_heads, n_patches + 1, n_patches + 1)

        weighted_values = attn_weight @ v # (num_samples, num_heads, n_patches + 1, head_dim)
        weighted_values = weighted_values.transpose(1, 2) # (num_samples, n_patches + 1, num_heads, head_dim)
        weighted_values = weighted_values.flatten(2) # (num_samples, n_patches + 1, dim)

        x = self.proj(weighted_values) # (num_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (num_samples, n_patches + 1, dim)
        
        return x

class MLP(nn.Module):
    """
    Multilayer Perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int, default=None
        Number of hidden features. If None then hidden_features=in_freatures

    out_features : int, default=None
        Number of output features. If None then out_features=in_freatures

    p : float, default=0.0
        Dropout probability.
        
    Attributes
    ----------
    fc1 : nn.Linear
        The first linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.         
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        p: float = 0.0
    ) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.
        """
        
        x = self.fc1(x) # (num_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (num_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (num_samples, n_patches + 1, out_features)
        x = self.drop(x) # (num_samples, n_patches + 1, out_features)

        return x

class Transformer(nn.Module):
    """
    Transformer block.

    Parameters
    ----------
    dim : int
        The input and output dimension of each feature token.

    num_heads : int, default=12
        Number of attention heads.

    mlp_ratio : float, default=4.0
        Determines the hidden dimension size of the `MLP` module with respect to `dim`.

    qkv_bias : bool, default=True
        If True then bias is included to the query, key, and value projections.

    qk_norm : bool, default=False
        If Tre then normalization is applied the query and key.

    proj_p, attn_p : float, default=0.0
        Dropout probability.
        
    Attributes
    ----------
    norm1, norm2 : nn.NormLayer
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int=12,
        mlp_ratio: float=4.0,
        qkv_bias: bool=True,
        qk_norm: bool=False,
        attn_p: float=0.0,
        proj_p: float=0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim = dim,
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            attn_p = attn_p,
            proj_p = proj_p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features = dim,
            hidden_features = int(dim * mlp_ratio),
            out_features = dim
        )

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """
    Pytorch implementation of Vision Transformer.

    Parameters
    ----------
    img_size : int, default=384
        Size of the image.

    patch_size : int, default=16
        Size of a patch.

    in_chans : int, default=3
        Number of image input channels.

    num_classes : int
        Number of classes.

    embed_dim : int, default=768
        Dimentionality of the token/patch embeddings in the Transformer.

    depth : int, default=12
        Number of Transformer blocks.

    num_heads : int, default=12
        Number of attention heads.

    mlp_ratio : float, default=4.0
        Ratio of mlp hidden dim to embedding dim.

    qkv_bias : bool, default=True
        If True then bias is included to the query, key, and value projections.

    qk_norm : bool, default=False
        If Tre then normalization is applied the query and key.

    proj_p: float, default=0.0
        Dropout probability for linear projection.
    
    attn_p : float, default=0.0
        Dropout probability for attention block.
        
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_embed : nn.Parameter
        Positional embedding of the class token + all the patches.
        It has `(num_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Transformer` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_p: float = 0.0,
        proj_p: float = 0.0
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(proj_p)
        self.blocks = nn.ModuleList(
            [
                Transformer(
                    dim = embed_dim,
                    num_heads = num_heads,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    qk_norm = qk_norm,
                    attn_p = attn_p,
                    proj_p = proj_p
                ) for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(num_samples, num_patches + 1, dim)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes.
            Shape `(num_samples, num_classes)`.
        """

        num_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(num_samples, -1, -1) # (num_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (num_samples, 1 + num_patches, embed_dim)
        x = x + self.pos_embed # (num_samples, 1 + num_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x

    def load_model(self, model_dir):
        """
        Load pre-trained model.

        Parameters
        ----------
        model_dir : str
            Path to pre-trained model.

        Returns
        -------
        Any
            Pre-trained Vision Transformer model.
        """

        return torch.load(model_dir)

    def load_checkpoint(self, checkpoint_dir):
        """
        Load pre-trained checkpoint.

        Parameters
        ----------
        checkpoint_dir : str
            Path to pre-trained checkpoint.
        """
        state_dict = torch.load(checkpoint_dir)
        self.load_state_dict(state_dict)
"""
Lightweight CLIP Module
Only use text encoder for state classification
"""

import torch
import torch.nn as nn


class LightweightCLIP(nn.Module):
    """
    Lightweight CLIP for state classification
    
    Only uses text encoder, vision encoder is removed.
    All parameters are frozen.
    
    Args:
        clip_model_path: Path to CLIP model
        out_channels: Output projection channels
    """
    
    def __init__(self, clip_model_path, out_channels):
        super().__init__()
        
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Load only text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
        
        # Freeze all parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Projection layer
        self.text_proj = nn.Linear(512, out_channels)
        
        # Cache for text features
        self._cached_features = None
        self._cached_prompts = None
    
    def encode_text(self, prompts):
        """
        Encode text prompts
        
        Args:
            prompts: List of text prompts ["no-damage", "minor-damage", ...]
            
        Returns:
            Text features [num_prompts, out_channels]
        """
        # Check cache
        if prompts == self._cached_prompts and self._cached_features is not None:
            return self._cached_features
        
        # Tokenize
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.text_encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            text_features = outputs.pooler_output  # [num_prompts, 512]
        
        # Project
        text_features = self.text_proj(text_features)  # [num_prompts, out_channels]
        
        # Cache
        self._cached_features = text_features
        self._cached_prompts = prompts
        
        return text_features
    
    def forward(self, prompts):
        """
        Forward pass
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text features [num_prompts, out_channels]
        """
        return self.encode_text(prompts)


class LearnableStateEmbeddings(nn.Module):
    """
    Learnable state embeddings (alternative to CLIP)
    
    Args:
        num_states: Number of state classes
        embed_dim: Embedding dimension
    """
    
    def __init__(self, num_states, embed_dim):
        super().__init__()
        
        self.state_embeddings = nn.Embedding(num_states, embed_dim)
        
        # Initialize
        nn.init.normal_(self.state_embeddings.weight, std=0.02)
    
    def get_weight(self):
        "Returns all state embeddings as [num_states, embed_dim] tensor."
        return self.state_embeddings.weight

    def forward(self, state_ids):
        """
        Forward pass
        
        Args:
            state_ids: State indices [num_states]
            
        Returns:
            State embeddings [num_states, embed_dim]
        """
        return self.state_embeddings(state_ids)


if __name__ == '__main__':
    # Test LearnableStateEmbeddings
    embeddings = LearnableStateEmbeddings(num_states=5, embed_dim=128)
    state_ids = torch.arange(5)
    features = embeddings(state_ids)
    print(f"State embeddings: {features.shape}")

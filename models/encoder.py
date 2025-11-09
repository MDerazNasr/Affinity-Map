#goal - turn a padded token sequence (shape [B,L]) int 0-20 into a fixed size embedding
#[B,D] (ex - D=128). this is what protoNets will use for distances
#a 1-D convolution (Conv1d) scans along the sequence looking for motifs (short patterns of amino acids that matter for function).
#we keep the strongest signal for each learned motif (pooling), compress to a final embedding (fixed-length vector), and normalize it so distances are meaningful.
# that final vector is what the few-shot algorithm (Prototypical Networks) will use to compare proteins.
import torch
import torch.nn as nn
import torch.nn.functional as F

#nn gives you layers; F gives you functional ops like relu, normalise

class ProteinEncoderCNN(nn.Module):
    def __init__(self, vocab_size=21, emb_dim=64, conv_dim=128, proj_dim=128):
        """
        CNN-based encoder for protein sequences.
        Args:
            vocab_size: number of amino-acid tokens (20 + 1 for PAD)
            emb_dim: size of token embeddings
            conv_dim: number of channels in convolutional layers
            proj_dim: size of the final embedding (output feature vector)
        """
        super().__init__()
        #nn.Embedding - turns token IDs into small vectors
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = 0) #standard pytorch module, we parametize sizes so you can tweak easily
        '''
        each Conv1d has a bunch of filters that slide along the sequence.
	    kernel_size=5 means “look at windows of 5 positions at a time” (like 5-grams).
	    padding keeps the length the same.
	    output channels = conv_dim=128 → you can think of it as learning 128 different motif detectors.
        '''
        self.conv1 = nn.Conv1d(emb_dim,  conv_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1) #128 -> 128

        self.dropout = nn.Dropout(0.1)

        #pooling - did this motif appear anywhere
        self.pool = nn.AdaptiveAvgPool1d(1)
        '''
        max pooling over the entire length picks the strongest activation for each of the 128 channels
        resulkt is a fixed-size vector per protein, independant of L
        shape is now (B, 128).
        why max? we care that a motif exists, not where it is. different proteins can have motifs in different positions
        '''

        #projection and normalization
        self.proj = nn.Linear(conv_dim, proj_dim) #128 -> 128
        '''
        final linear layer mixes the 128 channel repsosne into the fnial embedding psace ProtoNets will use
        L2 normalization makes cosine similarity and Euclidean distance behave nicely
        final shape: (B, proj_dim), e.g. (B, 128).
        this is your “protein fingerprint.”
        '''

    def forward(self, x):  # x: (B, L) integers 0..20 (0 = PAD)
        """
        Forward pass for a batch of tokenized protein sequences.

        Args:
            x: Tensor of shape (B, L) where each entry is an integer token ID
            (0 = PAD). Example: (batch=8, length=400)

        Returns:
            Tensor of shape (B, proj_dim), where each row is a normalized
            protein embedding vector suitable for distance-based learning.
        """
        # Step 1: Token embedding
        # x: (B, L) -> (B, L, E)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dtype != torch.long:
            x = x.long()
        assert x.dim() == 2, f"Encoder expects (B,L); got {tuple(x.shape)}"

        B, L = x.shape
        x = self.embedding(x)
        if x is None or x.dim() != 3:
            raise RuntimeError(
                f"Embedding failed: expected (B,L,E), got {None if x is None else tuple(x.shape)}. "
                "Check that self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=64, padding_idx=0)."
            )

        # Step 2: Permute for Conv1d input (channels-first)
        # (B, L, E) -> (B, E, L)
        x = x.transpose(1, 2)

        # Step 3: Convolutional motif extraction
        # Learn short local patterns (motifs) across the amino-acid sequence
        x = F.relu(self.conv1(x))       # (B, 128, L)
        x = F.relu(self.conv2(x))       # (B, 128, L)
        x = self.dropout(F.relu(self.conv3(x)))  # (B, 128, L)

        # Step 4: Global max pooling
        # Take the strongest activation per channel -> (B, 128, 1)
        x = self.pool(x).squeeze(-1)    # -> (B, 128)

        # Step 5: Projection to embedding space
        x = self.proj(x)                # (B, proj_dim)

        # Step 6: L2 normalization (important for ProtoNets)
        # Ensures distances are meaningful and consistent across proteins
        x = F.normalize(x, p=2, dim=-1, eps=1e-8) # (B, proj_dim)

        return x
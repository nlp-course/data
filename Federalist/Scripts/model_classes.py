"""
Language Model Classes for Federalist Papers Training

This module contains all the model architectures used in the labs:
- FFNNLM: N-gram Feedforward Neural Network Language Model
- RNNLM: Recurrent Neural Network Language Model  
- UALM: Uniform Attention Language Model
- ATTNLM: Learned Attention Language Model
- Transformer: Simple Transformer Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import copy

class BaseLM(nn.Module):
    """Base Language Model class with common functionality."""
    
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.pad_index = tokenizer.pad_token_id
        self.unk_index = tokenizer.unk_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For saving best model during training
        self.best_model = None
        self.best_val_loss = float('inf')
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)
    
    def tokens_to_ids(self, tokens):
        """Convert tokens to IDs using the tokenizer."""
        return [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
    
    def create_batches(self, token_ids, batch_size, sequence_length):
        """Create batches from token IDs."""
        batches = []
        for i in range(0, len(token_ids) - sequence_length, batch_size * sequence_length):
            batch_data = []
            batch_targets = []
            
            for j in range(batch_size):
                start_idx = i + j * sequence_length
                if start_idx + sequence_length + 1 <= len(token_ids):
                    # Input sequence
                    seq = token_ids[start_idx:start_idx + sequence_length]
                    # Target sequence (shifted by 1)
                    target = token_ids[start_idx + 1:start_idx + sequence_length + 1]
                    
                    batch_data.append(seq)
                    batch_targets.append(target)
            
            if batch_data:
                # Convert to tensors
                data_tensor = torch.tensor(batch_data, device=self.device)
                target_tensor = torch.tensor(batch_targets, device=self.device)
                batches.append((data_tensor, target_tensor))
        
        return batches
    
    def train_epoch(self, batches, optimizer):
        """Train for one epoch."""
        self.train()
        total_loss = 0
        
        for batch_data, batch_targets in tqdm(batches, desc="Training"):
            optimizer.zero_grad()
            loss = self.forward_step(batch_data, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(batches)
    
    def validate(self, batches):
        """Validate the model."""
        self.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in batches:
                loss = self.forward_step(batch_data, batch_targets)
                total_loss += loss.item()
        
        return total_loss / len(batches)
    
    def train_all(self, train_token_ids, val_token_ids, epochs=10, learning_rate=1e-3, 
                  batch_size=32, sequence_length=35):
        """Train the model with early stopping."""
        # Create batches
        train_batches = self.create_batches(train_token_ids, batch_size, sequence_length)
        val_batches = self.create_batches(val_token_ids, batch_size, sequence_length)
        
        print(f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_batches, optimizer)
            
            # Validation
            val_loss = self.validate(val_batches)
            
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.state_dict())
                print("  ✓ New best model saved")
        
        # Load best model
        if self.best_model is not None:
            self.load_state_dict(self.best_model)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
    
    def forward_step(self, data, target):
        """Forward step - to be implemented by subclasses."""
        raise NotImplementedError


class FFNNLM(BaseLM):
    """N-gram Feedforward Neural Network Language Model."""
    
    def __init__(self, n, tokenizer, embedding_size=128, hidden_size=128, dropout=0.0):
        super().__init__(tokenizer)
        self.n = n
        
        # Layers
        self.embed = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_index)
        self.sublayer1 = nn.Sequential(
            nn.Linear((n-1) * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.sublayer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.hidden2output = nn.Linear(hidden_size, self.vocab_size)
    
    def forward_step(self, text, target):
        batch_size, seq_len = text.shape
        
        # For FFNN, we need to create n-gram contexts
        # text is (batch_size, seq_len), target is (batch_size, seq_len)
        
        total_loss = 0
        count = 0
        
        for i in range(seq_len):
            # Create context of size n-1
            if i >= self.n - 1:
                # Use last n-1 tokens as context
                context = text[:, i-(self.n-1):i]  # (batch_size, n-1)
            else:
                # Pad with pad tokens
                pad_size = self.n - 1 - i
                padding = torch.full((batch_size, pad_size), self.pad_index, device=self.device)
                context = torch.cat([padding, text[:, :i]], dim=1)  # (batch_size, n-1)
            
            # Forward pass
            embeddings = self.embed(context)  # (batch_size, n-1, embedding_size)
            embeddings = embeddings.view(batch_size, -1)  # (batch_size, (n-1)*embedding_size)
            
            hidden1 = self.sublayer1(embeddings)  # (batch_size, hidden_size)
            hidden2 = self.sublayer2(hidden1)  # (batch_size, hidden_size)
            logits = self.hidden2output(hidden2)  # (batch_size, vocab_size)
            
            # Compute loss for this position
            targets_i = target[:, i]  # (batch_size,)
            loss = self.criterion(logits, targets_i)
            total_loss += loss
            count += 1
        
        return total_loss / count if count > 0 else total_loss
    
    def __call__(self, context):
        """Generate probability distribution for next token given context."""
        self.eval()
        with torch.no_grad():
            # Convert context to IDs if needed
            if isinstance(context, list):
                if isinstance(context[0], str):
                    context_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in context]
                else:
                    context_ids = context
            else:
                context_ids = context.tolist() if torch.is_tensor(context) else context
            
            # Ensure context is the right length
            if len(context_ids) < self.n - 1:
                # Pad with pad tokens
                context_ids = [self.pad_index] * (self.n - 1 - len(context_ids)) + context_ids
            else:
                # Take last n-1 tokens
                context_ids = context_ids[-(self.n - 1):]
            
            # Convert to tensor
            context_tensor = torch.tensor([context_ids], device=self.device)  # (1, n-1)
            
            # Forward pass
            embeddings = self.embed(context_tensor)  # (1, n-1, embedding_size)
            embeddings = embeddings.view(1, -1)  # (1, (n-1)*embedding_size)
            hidden1 = self.sublayer1(embeddings)  # (1, hidden_size)
            hidden2 = self.sublayer2(hidden1)  # (1, hidden_size)
            logits = self.hidden2output(hidden2)  # (1, vocab_size)
            probs = torch.softmax(logits, -1).view(-1)  # (vocab_size,)
            
            # Create distribution dictionary (compatible with labs)
            distribution = {}
            for i, prob in enumerate(probs):
                word = self.tokenizer.decode([i], clean_up_tokenization_spaces=True)
                distribution[word] = prob.item()
            
            return distribution


class RNNLM(BaseLM):
    """Recurrent Neural Network Language Model."""
    
    def __init__(self, tokenizer, embedding_size=128, hidden_size=128):
        super().__init__(tokenizer)
        
        # Layers
        self.embed = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_index)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.hidden2output = nn.Linear(hidden_size, self.vocab_size)
    
    def forward_step(self, text, target):
        batch_size, seq_len = text.shape
        
        # Embeddings
        embeddings = self.embed(text)  # (batch_size, seq_len, embedding_size)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(embeddings)  # (batch_size, seq_len, hidden_size)
        
        # Output projection
        logits = self.hidden2output(rnn_out)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)  # (batch_size * seq_len)
        
        loss = self.criterion(logits_flat, target_flat)
        return loss


class UALM(BaseLM):
    """Uniform Attention Language Model."""
    
    def __init__(self, tokenizer, embedding_size=128, hidden_size=128):
        super().__init__(tokenizer)
        
        # Layers
        self.embed = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_index)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.vocab_size)
    
    def forward_step(self, text, target):
        batch_size, seq_len = text.shape
        
        # Embeddings
        embeddings = self.embed(text)  # (batch_size, seq_len, embedding_size)
        
        # Uniform attention (simple average)
        # Create mask for padding
        mask = (text != self.pad_index).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_embeddings = embeddings * mask
        
        # Average over sequence length
        seq_lengths = mask.sum(dim=1)  # (batch_size, 1)
        avg_embeddings = masked_embeddings.sum(dim=1) / (seq_lengths + 1e-8)  # (batch_size, embedding_size)
        
        # Forward pass
        hidden = torch.relu(self.linear1(avg_embeddings))  # (batch_size, hidden_size)
        logits = self.linear2(hidden)  # (batch_size, vocab_size)
        
        # Expand logits to match target shape
        logits = logits.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)  # (batch_size * seq_len)
        
        loss = self.criterion(logits_flat, target_flat)
        return loss


class ATTNLM(BaseLM):
    """Learned Attention Language Model."""
    
    def __init__(self, tokenizer, embedding_size=128, hidden_size=128):
        super().__init__(tokenizer)
        
        # Layers
        self.embed = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_index)
        
        # Attention layers
        self.q_proj = nn.Linear(embedding_size, hidden_size)
        self.k_proj = nn.Linear(embedding_size, hidden_size)
        self.v_proj = nn.Linear(embedding_size, hidden_size)
        
        self.output_proj = nn.Linear(hidden_size, self.vocab_size)
        
        self.scale = math.sqrt(hidden_size)
    
    def forward_step(self, text, target):
        batch_size, seq_len = text.shape
        
        # Embeddings
        embeddings = self.embed(text)  # (batch_size, seq_len, embedding_size)
        
        # Attention computation
        queries = self.q_proj(embeddings)  # (batch_size, seq_len, hidden_size)
        keys = self.k_proj(embeddings)  # (batch_size, seq_len, hidden_size)
        values = self.v_proj(embeddings)  # (batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale  # (batch_size, seq_len, seq_len)
        
        # Create causal mask (prevent looking ahead)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # Create padding mask
        pad_mask = (text == self.pad_index).unsqueeze(1)  # (batch_size, 1, seq_len)
        scores.masked_fill_(pad_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, values)  # (batch_size, seq_len, hidden_size)
        
        # Output projection
        logits = self.output_proj(attended)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)  # (batch_size * seq_len)
        
        loss = self.criterion(logits_flat, target_flat)
        return loss


class Transformer(BaseLM):
    """Simple Transformer Language Model."""
    
    def __init__(self, tokenizer, embedding_size=128, hidden_size=128, num_heads=8, num_layers=2):
        super().__init__(tokenizer)
        
        # Layers
        self.embed = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_index)
        self.pos_embed = nn.Parameter(torch.randn(1000, embedding_size))  # Fixed positional embeddings
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(embedding_size, self.vocab_size)
    
    def forward_step(self, text, target):
        batch_size, seq_len = text.shape
        
        # Embeddings
        embeddings = self.embed(text)  # (batch_size, seq_len, embedding_size)
        
        # Add positional embeddings
        embeddings += self.pos_embed[:seq_len].unsqueeze(0)  # (batch_size, seq_len, embedding_size)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        
        # Create padding mask
        src_key_padding_mask = (text == self.pad_index)  # (batch_size, seq_len)
        
        # Transformer forward pass
        output = self.transformer(
            embeddings,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, embedding_size)
        
        # Output projection
        logits = self.output_proj(output)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)  # (batch_size * seq_len)
        
        loss = self.criterion(logits_flat, target_flat)
        return loss
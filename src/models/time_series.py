"""
PyTorch time series models for sales forecasting.
This module implements LSTM and Transformer models for time series forecasting.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import logging
import pickle
import time
import argparse
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('time_series')

# Import settings if available
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.settings import (
        COMBINED_DATA_FILE, MODELS_DIR, PYTORCH_FORECASTS_FILE, STATIC_DIR
    )
except ImportError:
    # Default paths for backward compatibility
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    COMBINED_DATA_FILE = os.path.join(ROOT_DIR, "combined_pizza_data.csv")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    PYTORCH_FORECASTS_FILE = os.path.join(ROOT_DIR, "pytorch_forecasts.csv")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")

# Make holidays optional for testing
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    logger.warning("'holidays' module not found. Some features will be limited.")


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch dataset for time series data
    """
    def __init__(self, X, y, weights=None):
        """
        Initialize time series dataset
        
        Args:
            X: Feature data
            y: Target data
            weights: Optional sample weights
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Include sample weights (default to 1.0 if not provided)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = torch.ones_like(self.y)
    
    def __len__(self):
        """Return dataset length"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        return self.X[idx], self.y[idx], self.weights[idx]


class LSTMModel(nn.Module):
    """
    Enhanced LSTM model for time series forecasting with complex architecture
    """
    def __init__(self, input_dim, hidden_dim=768, num_layers=8, output_dim=1, dropout=0.4):
        """
        Initialize enhanced LSTM model with more robust architecture
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (increased to 768)
            num_layers: Number of LSTM layers (increased to 8)
            output_dim: Output dimension
            dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        logger.info(f"Created enhanced LSTM model with {input_dim} input features")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Multi-stage input processing for better feature extraction
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),  # GELU activation often works better than ReLU for deep networks
            nn.Dropout(dropout * 0.5),  # Lighter dropout for input layer
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Deeper LSTM with residual connections and bidirectional options in deeper layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            # First half of layers are unidirectional, second half bidirectional for better context
            is_bidirectional = i >= num_layers // 2
            input_size = hidden_dim if i == 0 else hidden_dim * (2 if is_bidirectional else 1)
            output_size = hidden_dim // (2 if is_bidirectional else 1)  # Adjust size for bidirectional
            
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=output_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=is_bidirectional,
                    dropout=0  # Handle dropout separately
                )
            )
        
        # Enhanced layer normalization with more parameters
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * (2 if i >= num_layers // 2 else 1))
            for i in range(num_layers)
        ])
        
        # Graduated dropout (increases with depth)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout * min(1.0, 0.5 + i * 0.1))  # Increase dropout with depth
            for i in range(num_layers)
        ])
        
        # Skip connections at different levels (every other layer)
        self.use_skip_connection = [i > 0 and i % 2 == 0 for i in range(num_layers)]
        
        # Attention layer for focusing on important time steps
        attention_dim = hidden_dim * (2 if num_layers > 0 and (num_layers - 1) >= num_layers // 2 else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Wider and deeper output processing network
        fc_input_dim = hidden_dim * (2 if num_layers > 0 and (num_layers - 1) >= num_layers // 2 else 1)
        self.fc_input_dim = fc_input_dim  # Store for dynamic layer creation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Create fc_layers as ModuleList so we can dynamically adapt it
        self.fc_norm = nn.LayerNorm(fc_input_dim)  # Normalize before processing
        self.fc_layer1 = nn.Linear(fc_input_dim, hidden_dim)
        self.fc_gelu1 = nn.GELU()
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_gelu2 = nn.GELU()
        self.fc_dropout2 = nn.Dropout(dropout * 0.8)  # Slightly reduce dropout before final layer
        self.fc_norm2 = nn.LayerNorm(hidden_dim // 2)  # Extra normalization for stability
        self.fc_output = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the enhanced LSTM model
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Enhanced multi-stage input projection
        x_proj = self.input_projection(x)
        
        # Process through LSTM layers with advanced residual connections
        h_prev = x_proj
        skip_connections = []  # Store intermediate outputs for skip connections
        
        # Store the dimension of the final output for the attention layer
        final_output_dim = None
        
        # Initialize all layers with correct dimensions from the beginning
        # This prevents the need for runtime dimension adjustment
        
        for i, (lstm, norm, dropout) in enumerate(zip(self.lstm_layers, self.layer_norms, self.dropouts)):
            # Store skip connection if needed
            if i > 0 and i % 2 == 0 and i < len(self.lstm_layers) - 1:
                skip_connections.append(h_prev)
                
            # Initialize hidden states for this layer
            is_bidirectional = i >= self.num_layers // 2
            directions = 2 if is_bidirectional else 1
            hidden_size = self.hidden_dim // (2 if is_bidirectional else 1)
            
            h0 = torch.zeros(directions, batch_size, hidden_size).to(x.device)
            c0 = torch.zeros(directions, batch_size, hidden_size).to(x.device)
            
            # Process through LSTM
            try:
                # Check dimensions and ensure they match before processing
                expected_input_size = lstm.input_size
                current_input_size = h_prev.size(-1)
                
                if i == 0:
                    logger.info(f"LSTM input shape: {h_prev.size()}, expecting input_size={expected_input_size}")
                
                # Always ensure dimensions match before processing - this is crucial
                if current_input_size != expected_input_size:
                    # Create a proper linear projection to match dimensions
                    projection = nn.Linear(current_input_size, expected_input_size).to(x.device)
                    h_prev = projection(h_prev)
                    logger.info(f"Projected input from {current_input_size} to {expected_input_size}")
                    
                # Process through LSTM
                out, _ = lstm(h_prev, (h0, c0))
                
                # Verify output shape for debugging
                if i == 0:
                    logger.info(f"LSTM output shape after layer {i}: {out.shape}")
                    
            except RuntimeError as e:
                logger.error(f"LSTM layer {i} error: {e}")
                # Don't try to fix dimension issues dynamically - it's better to 
                # properly initialize the architecture from the start
                raise
            
            # Apply normalization and dropout with increasing intensity
            # Make sure the normalization layer matches the output dimension
            if out.size(-1) != norm.normalized_shape[0]:
                # Create a new LayerNorm with correct dimensions
                new_norm = nn.LayerNorm(out.size(-1)).to(x.device)
                out = new_norm(out)
            else:
                out = norm(out)
                
            out = dropout(out)
            
            # Apply skip connection if this is a skip layer
            if self.use_skip_connection[i] and len(skip_connections) > 0:
                # Match dimensions if needed
                skip = skip_connections.pop(0)
                if skip.size(-1) != out.size(-1):
                    # Use a proper linear projection to match dimensions
                    skip_projection = nn.Linear(skip.size(-1), out.size(-1)).to(x.device)
                    skip = skip_projection(skip)
                
                # Add residual connection
                out = out + skip
            
            # Apply residual connection if not the first layer and dimensions match
            elif i > 0 and h_prev.size(-1) == out.size(-1):
                out = out + h_prev
                
            h_prev = out
            
            # Store the dimension of the final output
            if i == len(self.lstm_layers) - 1:
                final_output_dim = out.size(-1)
        
        # Apply attention mechanism to focus on important time steps
        if self.num_layers > 0:  # Only if we have processed through at least one LSTM layer
            # Prepare for attention: [seq_len, batch, features]
            out_for_attn = out.permute(1, 0, 2)
            
            # Check if dimensions match between the attention layer and the input
            expected_dim = self.attention.embed_dim
            actual_dim = out_for_attn.size(-1)
            
            if actual_dim != expected_dim:
                # Create a projection layer to match dimensions
                projection = nn.Linear(actual_dim, expected_dim).to(x.device)
                out_for_attn = projection(out_for_attn)
                logger.info(f"Projected attention input from {actual_dim} to {expected_dim}")
            
            # Self-attention across the sequence dimension
            attn_out, _ = self.attention(out_for_attn, out_for_attn, out_for_attn)
            
            # Back to original shape: [batch, seq_len, features]
            attn_out = attn_out.permute(1, 0, 2)
            
            # Project attention output back to the original dimension if needed
            if attn_out.size(-1) != out.size(-1):
                # Create a projection layer to match dimensions for residual connection
                out_projection = nn.Linear(attn_out.size(-1), out.size(-1)).to(x.device)
                attn_out = out_projection(attn_out)
                logger.info(f"Projected attention output from {self.attention.embed_dim} to {out.size(-1)}")
                
            # Add residual connection
            out = out + attn_out
        
        # Get the last output with enhanced context
        out = out[:, -1, :]
        
        # Pass through enhanced final layers with dimension handling
        # Create a dynamic processing pipeline based on the actual input dimensions
        try:
            # First, check dimensions and normalize
            if out.size(-1) != self.fc_norm.normalized_shape[0]:
                logger.info(f"Creating dynamic fc_norm for size {out.size(-1)}")
                dynamic_norm = nn.LayerNorm(out.size(-1)).to(out.device)
                out = dynamic_norm(out)
            else:
                out = self.fc_norm(out)
                
            # Process through first layer with dynamic adaptation if needed
            if out.size(-1) != self.fc_layer1.in_features:
                logger.info(f"Creating dynamic fc_layer1 from {out.size(-1)} to {self.hidden_dim}")
                dynamic_layer1 = nn.Linear(out.size(-1), self.hidden_dim).to(out.device)
                out = dynamic_layer1(out)
            else:
                out = self.fc_layer1(out)
                
            # Continue processing
            out = self.fc_gelu1(out)
            out = self.fc_dropout1(out)
            out = self.fc_layer2(out)
            out = self.fc_gelu2(out)
            out = self.fc_dropout2(out)
            out = self.fc_norm2(out)
            out = self.fc_output(out)
        except Exception as e:
            logger.error(f"Error in FC processing: {e}")
            # Fallback to a simple linear layer if needed
            out = nn.Linear(out.size(-1), self.output_dim).to(out.device)(out)
        
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Transformer model has been commented out
# class TransformerModel(nn.Module):
#     """
#     Enhanced Transformer model for time series forecasting with complex architecture
#     """
#     def __init__(self, input_dim, d_model=768, nhead=12, num_layers=12, output_dim=1, dropout=0.3):
#         """
#         Initialize enhanced transformer model
#         
#         Args:
#             input_dim: Input feature dimension
#             d_model: Model dimension (increased to 768)
#             nhead: Number of attention heads (increased to 12)
#             num_layers: Number of transformer layers (increased to 12)
#             output_dim: Output dimension
#             dropout: Dropout rate
#         """
#         super(TransformerModel, self).__init__()
#         
#         # Enhanced multi-stage input processing
#         self.input_projection = nn.Sequential(
#             nn.Linear(input_dim, d_model // 2),
#             nn.LayerNorm(d_model // 2),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(d_model // 2, d_model),
#             nn.LayerNorm(d_model)
#         )
#         
#         # Enhanced positional encoding with learnable parameters
#         self.pos_encoder = nn.Sequential(
#             PositionalEncoding(d_model, dropout=dropout * 0.5, max_len=5000),
#             nn.Linear(d_model, d_model),  # Learnable transformation of positional encoding
#             nn.LayerNorm(d_model),
#             nn.Dropout(dropout * 0.3)
#         )
#         
#         # Create stacked encoder layers with varying complexity
#         self.encoder_layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         
#         for i in range(num_layers):
#             # Increase feedforward dimension and heads with depth for more capacity
#             layer_d_model = d_model
#             layer_nhead = nhead + (i // 3) * 2  # Gradually increase heads
#             layer_dim_ff = d_model * (4 + (i // 4))  # Gradually increase feedforward dimension
#             layer_dropout = dropout * min(1.5, 0.7 + i * 0.05)  # Gradually increase dropout
#             
#             # Create encoder layer with enhanced parameters
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=layer_d_model,
#                 nhead=layer_nhead,
#                 dim_feedforward=layer_dim_ff,
#                 dropout=layer_dropout,
#                 activation='gelu',
#                 batch_first=True,
#                 norm_first=True  # Apply normalization before attention for better stability
#             )
#             
#             self.encoder_layers.append(encoder_layer)
#             self.norms.append(nn.LayerNorm(layer_d_model))
#         
#         # Final normalization layer
#         self.final_norm = nn.LayerNorm(d_model)
#         
#         # Enhanced attention pooling for sequence aggregation
#         self.attention_pooling = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=nhead,
#             dropout=dropout,
#             batch_first=True
#         )
#         
#         # Learnable query for attention pooling
#         self.query = nn.Parameter(torch.randn(1, 1, d_model))
#         
#         # Enhanced output processing with residual connections
#         self.output_projection = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model // 2),
#             nn.LayerNorm(d_model // 2),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.8),
#             nn.Linear(d_model // 2, d_model // 4),
#             nn.LayerNorm(d_model // 4),
#             nn.GELU(),
#             nn.Linear(d_model // 4, output_dim)
#         )
#         
#         # Additional specific layers for time series
#         self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
#         self.time_feature_gate = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Sigmoid()
#         )
#         
#     def forward(self, x):
#         """
#         Forward pass through the enhanced transformer model
#         
#         Args:
#             x: Input tensor [batch_size, seq_len, input_dim]
#             
#         Returns:
#             Output tensor [batch_size, output_dim]
#         """
#         batch_size, seq_len, _ = x.shape
#         
#         # Enhanced multi-stage input processing
#         x = self.input_projection(x)
#         
#         # Apply positional encoding with learnable transformation
#         x = self.pos_encoder(x)
#         
#         # Apply temporal convolution for local feature extraction
#         x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
#         x_conv = self.temporal_conv(x_conv)
#         x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
#         
#         # Combine with gating mechanism
#         gate = self.time_feature_gate(x)
#         x = x * gate + x_conv * (1 - gate)
#         
#         # Process through transformer layers with residual connections
#         attn_weights = []
#         
#         # Pass through each transformer encoder layer individually
#         for i, (encoder_layer, norm) in enumerate(zip(self.encoder_layers, self.norms)):
#             # Store residual
#             residual = x if i % 2 == 0 else None
#             
#             # Pass through encoder layer
#             x = encoder_layer(x)
#             
#             # Apply additional normalization
#             x = norm(x)
#             
#             # Apply residual connection every other layer
#             if residual is not None:
#                 x = x + residual * 0.1  # Scaled residual connection
#         
#         # Final normalization
#         x = self.final_norm(x)
#         
#         # Enhanced sequence aggregation with attention pooling
#         # Expand query to batch size
#         query = self.query.expand(batch_size, -1, -1)
#         
#         # Apply attention pooling - query attends to the sequence
#         pooled_x, attention = self.attention_pooling(query, x, x)
#         pooled_x = pooled_x.squeeze(1)  # [batch_size, d_model]
#         
#         # Combine attention pooling with last token representation
#         last_token = x[:, -1, :]
#         combined = pooled_x + last_token
#         
#         # Apply enhanced output projection
#         output = self.output_projection(combined)
#         
#         return output


class ModelEnsemble:
    """
    Ensemble of time series models
    """
    def __init__(self, models, weights=None):
        """
        Initialize an ensemble of time series models
        
        Args:
            models: List of trained PyTorch models
            weights: Optional list of weights for each model (default: equal weights)
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]
    
    def predict(self, X, device='cpu'):
        """
        Make predictions with the ensemble
        
        Args:
            X: Input features
            device: PyTorch device to use
            
        Returns:
            Weighted average of predictions
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(self.models):
            try:
                model.eval()  # Set to evaluation mode
                with torch.no_grad():
                    # Handle dimension mismatch for the future sequences
                    expected_features = 0
                    if hasattr(model, 'input_projection'):
                        if hasattr(model.input_projection, '__getitem__'):
                            expected_features = model.input_projection[0].in_features
                        elif hasattr(model.input_projection, 'in_features'):
                            expected_features = model.input_projection.in_features
                    elif hasattr(model, 'input_proj'):
                        expected_features = model.input_proj.in_features
                        
                    if len(X.shape) == 3 and expected_features > 0 and X.shape[2] != expected_features:
                        # Check for feature dimension mismatch
                        logger.info(f"Model {i}: Expected {expected_features} features, got {X.shape[2]}")
                        # Create a projection layer to handle the feature dimension mismatch
                        projection = nn.Linear(X.shape[2], expected_features).to(device)
                        X_projected = projection(X_tensor)
                        pred = model(X_projected).cpu().numpy()
                        predictions.append(pred)
                        continue
                    pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error in model {i} prediction: {e}")
                continue
        
        if not predictions:
            # Fallback: create a simple prediction model on the fly
            logger.warning("No ensemble models could make predictions. Creating fallback model.")
            try:
                # Create a simple linear model that can handle the input shape
                input_size = X_tensor.size(-1)
                fallback_model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(device)
                
                # Make prediction
                with torch.no_grad():
                    fallback_pred = fallback_model(X_tensor).cpu().numpy()
                return fallback_pred
            except Exception as e:
                logger.error(f"Fallback prediction failed: {e}")
                # Last resort: return zeros - ensure shape is compatible with scaler
                if len(X.shape) == 3:  # For 3D tensors (batch, seq_len, features)
                    return np.zeros((X.shape[0]))
                else:
                    return np.zeros((X.shape[0], 1))
        
        # Weighted average of predictions
        weighted_pred = np.zeros_like(predictions[0])
        # Recalculate weights based on available models
        weight_sum = sum(self.weights[:len(predictions)])
        adjusted_weights = [w / weight_sum for w in self.weights[:len(predictions)]]
        
        for i, pred in enumerate(predictions):
            weighted_pred += pred * adjusted_weights[i]
            
        return weighted_pred


def prepare_time_series_features(df, seq_length=28):
    """
    Prepare features for time series models
    
    Args:
        df: Input DataFrame
        seq_length: Sequence length for time series
        
    Returns:
        Dictionary with processed time series data
    """
    logger.info(f"Preparing time series features with sequence length {seq_length}")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Process each store-item combination separately
    store_items = df[['Store_Id', 'Item']].drop_duplicates().values
    
    # Storage for all sequences
    all_X = []
    all_y = []
    all_dates = []
    all_store_ids = []
    all_items = []
    all_weights = []
    all_scalers = {}
    
    for store_id, item in store_items:
        logger.debug(f"Processing Store {store_id}, Item {item}")
        
        # Filter data for this store-item combination
        item_df = df[(df['Store_Id'] == store_id) & (df['Item'] == item)].sort_values('Date')
        
        # Skip if not enough data
        if len(item_df) <= seq_length:
            logger.debug(f"Skipping Store {store_id}, Item {item}: insufficient data ({len(item_df)} rows)")
            continue
        
        # Prepare features (both calendar and statistical)
        features = []
        
        # Sales data (target)
        sales = item_df['Sales'].values
        
        # Calendar features - encoded numerically
        # Enhanced seasonality for better handling of sparse data
        # Day of week - multiple harmonics for more detailed daily patterns
        day_of_week = item_df['Day_Of_Week'].values
        # First harmonic - basic weekly cycle
        features.append(np.sin(day_of_week * (2 * np.pi / 7)))
        features.append(np.cos(day_of_week * (2 * np.pi / 7)))
        # Second harmonic - captures twice-weekly patterns
        features.append(np.sin(day_of_week * (4 * np.pi / 7)))
        features.append(np.cos(day_of_week * (4 * np.pi / 7)))
        
        # Month - multiple harmonics for seasonal patterns
        month = item_df['Month'].values
        # Annual cycle
        features.append(np.sin(month * (2 * np.pi / 12)))
        features.append(np.cos(month * (2 * np.pi / 12)))
        # Quarterly cycle - captures seasonal changes
        features.append(np.sin(month * (2 * np.pi / 3)))
        features.append(np.cos(month * (2 * np.pi / 3)))
        
        # Is weekend flag
        is_weekend = (day_of_week >= 5).astype(float)
        features.append(is_weekend)
        
        # Add holiday flag if available
        if HOLIDAYS_AVAILABLE:
            us_holidays = holidays.US()
            holiday_flag = item_df['Date'].apply(lambda d: d in us_holidays).astype(float).values
            features.append(holiday_flag)
            
        # Add price info if available
        if 'Price' in item_df.columns:
            # Normalize price
            scaler = StandardScaler()
            price_normalized = scaler.fit_transform(item_df['Price'].values.reshape(-1, 1)).flatten()
            features.append(price_normalized)
        
        # Add weather if available (one-hot encoded)
        if 'Weather' in item_df.columns:
            # Get unique weather values
            weather_types = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
            for weather_type in weather_types:
                weather_flag = (item_df['Weather'] == weather_type).astype(float).values
                features.append(weather_flag)
        
        # Add promotion flag if available
        if 'Promotion' in item_df.columns:
            promotion_flag = item_df['Promotion'].astype(float).values
            features.append(promotion_flag)
            
        # Stock level features if available
        if 'Stock_Level' in item_df.columns:
            # Normalize stock level
            stock_scaler = StandardScaler()
            stock_normalized = stock_scaler.fit_transform(
                item_df['Stock_Level'].values.reshape(-1, 1)).flatten()
            features.append(stock_normalized)
            
        # Combine all features
        X_raw = np.column_stack(features)
        
        # Scale sales data for this item
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()
        
        # Create sequences with improved handling for sparse data
        sequences_X = []
        sequences_y = []
        sequence_dates = []
        sequence_weights = []
        
        # Calculate additional statistics for better sparse data handling
        sales_mean = np.mean(sales)
        sales_std = np.std(sales)
        zero_ratio = np.sum(sales == 0) / len(sales)
        
        for i in range(len(item_df) - seq_length):
            # Sequence of features
            sequences_X.append(X_raw[i:i+seq_length])
            # Target is the next day's sales
            sequences_y.append(y_scaled[i+seq_length])
            # Keep track of the prediction date
            sequence_dates.append(item_df['Date'].iloc[i+seq_length])
            
            # Enhanced weighting scheme for sparse data
            days_ago = (item_df['Date'].max() - item_df['Date'].iloc[i+seq_length]).days
            recency_weight = np.exp(-0.01 * days_ago)  # Exponential decay for recency
            
            # Add more weight to non-zero sales data points (helps with sparse data)
            target_sales = item_df['Sales'].iloc[i+seq_length]
            sales_value_weight = 1.0
            if target_sales > 0:
                # Higher weight for non-zero values in sparse datasets
                if zero_ratio > 0.5:  # If dataset is sparse (>50% zeros)
                    sales_value_weight = 2.0 + 2.0 * zero_ratio
                # Higher weight for unusual values that deviate from the mean
                if sales_std > 0 and target_sales > sales_mean + sales_std:
                    sales_value_weight += 1.0
                    
            # Combine weights
            weight = recency_weight * sales_value_weight
            sequence_weights.append(weight)
        
        # Convert to arrays
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        # Add to the overall collections
        all_X.append(sequences_X)
        all_y.append(sequences_y)
        all_dates.extend(sequence_dates)
        all_weights.append(sequence_weights)
        all_store_ids.extend([store_id] * len(sequences_y))
        all_items.extend([item] * len(sequences_y))
        
        # Store scaler for this item
        all_scalers[(store_id, item)] = scaler
    
    # Combine all sequences
    if not all_X:
        logger.error("No valid sequences could be created. Check data and sequence length.")
        return None
        
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    weights_combined = np.hstack(all_weights)
    
    # Create a mapping from index to metadata
    index_to_meta = {
        i: {'date': date, 'store_id': store, 'item': item} 
        for i, (date, store, item) in enumerate(zip(all_dates, all_store_ids, all_items))
    }
    
    logger.info(f"Created {len(X_combined)} sequences with {X_combined.shape[2]} features")
    
    return {
        'X': X_combined, 
        'y': y_combined, 
        'weights': weights_combined,
        'index_to_meta': index_to_meta,
        'scalers': all_scalers,
        'seq_length': seq_length,
        'feature_dim': X_combined.shape[2]
    }


def train_time_series_model(data, model_type='lstm', epochs=30, batch_size=64, 
                           learning_rate=0.001, device=None, weight_decay=1e-4, dropout=0.3):
    """
    Train time series model with the given data
    
    Args:
        data: Dictionary with processed time series data
        model_type: Model type ('lstm' only - transformer has been removed)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: PyTorch device
        
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Prepare data loaders
    X, y, weights = data['X'], data['y'], data['weights']
    feature_dim = data['feature_dim']
    
    # Split into train/validation
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(indices) * 0.8)  # 80% train, 20% validation
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X[train_indices], y[train_indices], weights[train_indices])
    val_dataset = TimeSeriesDataset(X[val_indices], y[val_indices], weights[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model with proper configuration for backward compatibility
    if model_type == 'lstm':
        try:
            # First try to create the enhanced model
            model = LSTMModel(input_dim=feature_dim, hidden_dim=768, num_layers=8).to(device)
            # Apply custom dropout settings to existing layers
            for layer in model.dropouts:
                layer.p = dropout  # Apply the dropout rate from function parameter
            logger.info(f"Created enhanced LSTM model with {feature_dim} input features")
        except Exception as e:
            # Fall back to simpler architecture for compatibility
            logger.warning(f"Error creating enhanced LSTM model: {e}. Falling back to legacy architecture.")
            model = nn.Sequential(
                nn.LSTM(feature_dim, 512, 2, batch_first=True),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            ).to(device)
    # Transformer model has been commented out
    # elif model_type == 'transformer':
    #     try:
    #         # First try to create the enhanced model
    #         model = TransformerModel(input_dim=feature_dim, d_model=768, dropout=dropout).to(device)
    #         logger.info(f"Created enhanced Transformer model with {feature_dim} input features")
    #     except Exception as e:
    #         # Fall back to simpler architecture for compatibility
    #         logger.warning(f"Error creating enhanced Transformer model: {e}. Falling back to basic architecture.")
    #         encoder_layer = nn.TransformerEncoderLayer(
    #             d_model=512, 
    #             nhead=8, 
    #             dim_feedforward=2048,
    #             dropout=dropout,
    #             activation='gelu',
    #             batch_first=True
    #         )
    #         
    #         # Create a more robust model with proper output handling
    #         class SimpleTransformerModel(nn.Module):
    #             def __init__(self, feature_dim, d_model=512, output_dim=1):
    #                 super(SimpleTransformerModel, self).__init__()
    #                 self.input_proj = nn.Linear(feature_dim, d_model)
    #                 # Add input_projection attribute for compatibility
    #                 self.input_projection = self.input_proj
    #                 self.encoder_layer = encoder_layer
    #                 self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
    #                 self.output_proj = nn.Linear(d_model, output_dim)
    #                 
    #             def forward(self, x):
    #                 # Project to d_model dimension
    #                 x = self.input_proj(x)
    #                 
    #                 # Process with transformer
    #                 x = self.transformer_encoder(x)
    #                 
    #                 # Get output from last time step
    #                 x = x[:, -1, :]
    #                 
    #                 # Project to output dimension
    #                 x = self.output_proj(x)
    #                 
    #                 return x
    #         
    #         model = SimpleTransformerModel(feature_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    logger.info(f"Initialized {model_type.upper()} model with {feature_dim} input features")
    
    # Define loss and optimizer with regularization for overfitting prevention
    criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply sample weights
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # AdamW with weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Train model with enhanced early stopping
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    # Track consecutive loss increases for more robust early stopping
    consecutive_increases = 0
    max_consecutive_increases = 3  # Stop if loss increases 3 times in a row
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch).squeeze()
            
            # Compute weighted loss
            loss = criterion(outputs, y_batch)
            weighted_loss = (loss * w_batch).mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)  # More aggressive gradient clipping to prevent overfitting
            optimizer.step()
            
            train_losses.append(weighted_loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)
                
                outputs = model(X_batch).squeeze()
                
                loss = criterion(outputs, y_batch)
                weighted_loss = (loss * w_batch).mean()
                
                val_losses.append(weighted_loss.item())
        
        # Calculate average losses
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model with enhanced early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
            consecutive_increases = 0  # Reset consecutive increases counter
            logger.info(f"New best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
            # Track if validation loss is consistently increasing (sign of overfitting)
            if epoch > 0 and val_loss > val_losses[-2] if len(val_losses) > 1 else 0:
                consecutive_increases += 1
            else:
                consecutive_increases = 0
            
        # Early stopping - two conditions:
        # 1. Patience exceeded (no improvement for a while)
        # 2. Validation loss consistently increasing (clear sign of overfitting)
        if patience_counter >= patience or consecutive_increases >= max_consecutive_increases:
            logger.info(f"Early stopping after {epoch+1} epochs" + 
                      (" due to consistently increasing validation loss" if consecutive_increases >= max_consecutive_increases else ""))
            break
    
    # Load best model weights
    if best_model is not None:
        model.load_state_dict(best_model)
        
    logger.info("Training complete")
    return model


def save_model(model, store_id, item, model_type='lstm'):
    """
    Save time series model to disk
    
    Args:
        model: Trained model
        store_id: Store ID
        item: Item ID
        model_type: Model type
        
    Returns:
        Path to saved model
    """
    # Create directory if needed
    ts_model_dir = os.path.join(MODELS_DIR, "time_series")
    os.makedirs(ts_model_dir, exist_ok=True)
    
    # Create filename
    filename = f"model_{store_id}_{item}.pt"
    filepath = os.path.join(ts_model_dir, filename)
    
    # Save model
    try:
        torch.save(model.state_dict(), filepath)
        logger.info(f"Saved model to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        return None


def save_scaler(scaler, store_id, item):
    """
    Save scaler to disk
    
    Args:
        scaler: StandardScaler object
        store_id: Store ID
        item: Item ID
        
    Returns:
        Path to saved scaler
    """
    # Create directory if needed
    ts_model_dir = os.path.join(MODELS_DIR, "time_series")
    os.makedirs(ts_model_dir, exist_ok=True)
    
    # Create filename
    filename = f"scaler_{store_id}_{item}.npz"
    filepath = os.path.join(ts_model_dir, filename)
    
    # Save scaler
    try:
        np.savez(filepath, mean=scaler.mean_, var=scaler.var_)
        logger.info(f"Saved scaler to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving scaler to {filepath}: {e}")
        return None


def load_model(store_id, item, feature_dim, model_type='lstm', device=None):
    """
    Load time series model from disk
    
    Args:
        store_id: Store ID
        item: Item ID
        feature_dim: Input feature dimension
        model_type: Model type (only 'lstm' supported now)
        device: PyTorch device
        
    Returns:
        Loaded model or None if not found
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create filename
    filename = f"model_{store_id}_{item}.pt"
    filepath = os.path.join(MODELS_DIR, "time_series", filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"No saved model found for Store {store_id}, Item {item}")
        return None
    
    try:
        # First try to load the state dict to inspect its keys
        state_dict = torch.load(filepath, map_location=device)
        
        # Check if state dict matches current model architecture or legacy architecture
        is_legacy_lstm = any(key.startswith(('lstm.', 'fc.')) for key in state_dict.keys())
        
        # Only load LSTM models now
        if model_type == 'lstm':
            if is_legacy_lstm:
                # Enhanced Legacy LSTM model with more robust architecture while maintaining compatibility
                class LegacyLSTMModel(nn.Module):
                    def __init__(self, input_dim, hidden_dim=512, num_layers=2, output_dim=1):
                        super(LegacyLSTMModel, self).__init__()
                        self.hidden_dim = hidden_dim
                        self.num_layers = num_layers
                        
                        # Input projection for feature transformation (maintains legacy structure)
                        self.input_projection = nn.Linear(input_dim, input_dim)
                        
                        # Main LSTM - preserve the same naming to match legacy weights
                        self.lstm = nn.LSTM(
                            input_dim, hidden_dim, num_layers, 
                            batch_first=True,
                            bidirectional=False,  # Match legacy structure
                            dropout=0.3 if num_layers > 1 else 0
                        )
                        
                        # Enhanced output layers while keeping the original fc name for compatibility
                        self.fc = nn.Sequential(
                            nn.LayerNorm(hidden_dim),  # Add normalization for stability
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.4),  # Increased dropout for better regularization
                            nn.Linear(hidden_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(hidden_dim // 2, output_dim)
                        )
                    
                    def forward(self, x):
                        # Apply input projection
                        x = self.input_projection(x)
                        
                        # Initialize hidden states with learned initial values
                        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                        
                        # Process through LSTM
                        out, _ = self.lstm(x, (h0, c0))
                        
                        # Only take output from the last time step
                        out = out[:, -1, :]
                        
                        # Apply enhanced FC layers while maintaining backward compatibility
                        out = self.fc(out)
                        return out
                
                model = LegacyLSTMModel(feature_dim).to(device)
                logger.info(f"Using legacy LSTM model architecture for Store {store_id}, Item {item}")
            else:
                model = LSTMModel(feature_dim).to(device)
        else:
            raise ValueError(f"Unknown or unsupported model type: {model_type} (only 'lstm' is supported)")
        
        # Load model weights
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        logger.info(f"Loaded model from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        return None


def load_scaler(store_id, item):
    """
    Load scaler from disk
    
    Args:
        store_id: Store ID
        item: Item ID
        
    Returns:
        Loaded scaler or None if not found
    """
    # Create filename
    filename = f"scaler_{store_id}_{item}.npz"
    filepath = os.path.join(MODELS_DIR, "time_series", filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"No saved scaler found for Store {store_id}, Item {item}")
        return None
    
    # Load scaler
    try:
        scaler_data = np.load(filepath)
        scaler = StandardScaler()
        scaler.mean_ = scaler_data['mean']
        scaler.var_ = scaler_data['var']
        scaler.scale_ = np.sqrt(scaler.var_)
        logger.info(f"Loaded scaler from {filepath}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler from {filepath}: {e}")
        return None


def create_future_features(df, store_id, item, last_date, days_to_forecast, seq_length=28):
    """
    Create feature sequences for future forecasting
    
    Args:
        df: Input DataFrame
        store_id: Store ID
        item: Item ID
        last_date: Last date in the data
        days_to_forecast: Number of days to forecast
        seq_length: Sequence length
        
    Returns:
        DataFrame with future features
    """
    logger.info(f"Creating future features for Store {store_id}, Item {item}, {days_to_forecast} days")
    
    # Filter data for this store-item
    item_df = df[(df['Store_Id'] == store_id) & (df['Item'] == item)].sort_values('Date')
    
    if len(item_df) < seq_length:
        logger.error(f"Insufficient data for Store {store_id}, Item {item}: need at least {seq_length} days")
        return None
    
    # Get the most recent sequence
    recent_data = item_df.iloc[-seq_length:].copy()
    
    # Generate future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_forecast)
    
    # Create future dataframe with calendar features
    future_rows = []
    
    for date in future_dates:
        # Basic calendar features
        day_of_week = date.dayofweek
        month = date.month
        
        # Copy the most recent values for other features
        latest_values = recent_data.iloc[-1].copy()
        
        # Create the future row
        future_row = {
            'Date': date,
            'Store_Id': store_id,
            'Item': item,
            'Product': latest_values['Product'],
            'Day_Of_Week': day_of_week,
            'Month': month,
        }
        
        # Add price (assume constant)
        if 'Price' in latest_values:
            future_row['Price'] = latest_values['Price']
            
        # Add weather (assume normal)
        if 'Weather' in latest_values:
            future_row['Weather'] = 'Normal'
            
        # Add promotion (assume no promotion)
        if 'Promotion' in latest_values:
            future_row['Promotion'] = 0
            
        # Add stock level (assume constant)
        if 'Stock_Level' in latest_values:
            future_row['Stock_Level'] = latest_values['Stock_Level']
            
        future_rows.append(future_row)
    
    future_df = pd.DataFrame(future_rows)
    
    # Prepare features (same as in training)
    features = []
    
    # Calendar features
    day_of_week = future_df['Day_Of_Week'].values
    features.append(np.sin(day_of_week * (2 * np.pi / 7)))
    features.append(np.cos(day_of_week * (2 * np.pi / 7)))
    
    month = future_df['Month'].values
    features.append(np.sin(month * (2 * np.pi / 12)))
    features.append(np.cos(month * (2 * np.pi / 12)))
    
    is_weekend = (day_of_week >= 5).astype(float)
    features.append(is_weekend)
    
    # Add holiday flag if available
    if HOLIDAYS_AVAILABLE:
        us_holidays = holidays.US()
        holiday_flag = future_df['Date'].apply(lambda d: d in us_holidays).astype(float).values
        features.append(holiday_flag)
        
    # Add price info if available
    if 'Price' in future_df.columns:
        # Use the same mean/std from training
        mean_price = item_df['Price'].mean()
        std_price = item_df['Price'].std()
        if std_price == 0:
            std_price = 1.0
        price_normalized = ((future_df['Price'].values - mean_price) / std_price)
        features.append(price_normalized)
    
    # Add weather if available (one-hot encoded)
    if 'Weather' in future_df.columns:
        weather_types = ['Normal', 'Heavy Rain', 'Snow', 'Storm']
        for weather_type in weather_types:
            weather_flag = (future_df['Weather'] == weather_type).astype(float).values
            features.append(weather_flag)
    
    # Add promotion flag if available
    if 'Promotion' in future_df.columns:
        promotion_flag = future_df['Promotion'].astype(float).values
        features.append(promotion_flag)
        
    # Stock level features if available
    if 'Stock_Level' in future_df.columns:
        # Use the same mean/std from training
        mean_stock = item_df['Stock_Level'].mean()
        std_stock = item_df['Stock_Level'].std()
        if std_stock == 0:
            std_stock = 1.0
        stock_normalized = ((future_df['Stock_Level'].values - mean_stock) / std_stock)
        features.append(stock_normalized)
            
    # Combine all features
    X_future = np.column_stack(features)
    
    # Get historical feature sequence
    historical_features = []
    
    # Process historical data with the same feature engineering
    hist_day_of_week = recent_data['Day_Of_Week'].values
    historical_features.append(np.sin(hist_day_of_week * (2 * np.pi / 7)))
    historical_features.append(np.cos(hist_day_of_week * (2 * np.pi / 7)))
    
    hist_month = recent_data['Month'].values
    historical_features.append(np.sin(hist_month * (2 * np.pi / 12)))
    historical_features.append(np.cos(hist_month * (2 * np.pi / 12)))
    
    hist_is_weekend = (hist_day_of_week >= 5).astype(float)
    historical_features.append(hist_is_weekend)
    
    # Add holiday flag if available
    if HOLIDAYS_AVAILABLE:
        hist_holiday_flag = recent_data['Date'].apply(lambda d: d in us_holidays).astype(float).values
        historical_features.append(hist_holiday_flag)
        
    # Add price info if available
    if 'Price' in recent_data.columns:
        hist_price_normalized = ((recent_data['Price'].values - mean_price) / std_price)
        historical_features.append(hist_price_normalized)
    
    # Add weather if available (one-hot encoded)
    if 'Weather' in recent_data.columns:
        for weather_type in weather_types:
            hist_weather_flag = (recent_data['Weather'] == weather_type).astype(float).values
            historical_features.append(hist_weather_flag)
    
    # Add promotion flag if available
    if 'Promotion' in recent_data.columns:
        hist_promotion_flag = recent_data['Promotion'].astype(float).values
        historical_features.append(hist_promotion_flag)
        
    # Stock level features if available
    if 'Stock_Level' in recent_data.columns:
        hist_stock_normalized = ((recent_data['Stock_Level'].values - mean_stock) / std_stock)
        historical_features.append(hist_stock_normalized)
    
    # Create the historical feature matrix
    X_hist = np.column_stack(historical_features)
    
    # For the first forecast, use the historical sequence
    X_sequence = X_hist.copy()
    
    # Create a list to store all forecast sequences
    forecast_sequences = []
    
    # For each day in the forecast period
    for i in range(days_to_forecast):
        # Store the current sequence
        forecast_sequences.append(X_sequence.copy())
        
        # Update the sequence by removing the oldest day and adding the new forecast day
        X_sequence = np.vstack((X_sequence[1:], X_future[i:i+1]))
    
    return {
        'X_sequences': np.array(forecast_sequences),
        'future_df': future_df
    }


def forecast_time_series(df, days_to_forecast=30, device=None):
    """
    Generate time series forecasts for all store-item combinations
    
    Args:
        df: Input DataFrame
        days_to_forecast: Number of days to forecast
        device: PyTorch device
        
    Returns:
        DataFrame with forecasts
    """
    logger.info(f"Generating time series forecasts for {days_to_forecast} days")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Get last date in the data
    last_date = df['Date'].max()
    
    # Get unique store-item combinations
    store_items = df[['Store_Id', 'Item']].drop_duplicates().values
    
    # Create model info dataframe
    model_info_data = []
    
    # Generate forecasts for each store-item combination
    all_forecasts = []
    
    for store_id, item in store_items:
        logger.info(f"Forecasting for Store {store_id}, Item {item}")
        
        # Get item details
        item_df = df[(df['Store_Id'] == store_id) & (df['Item'] == item)]
        if len(item_df) == 0:
            logger.warning(f"No data found for Store {store_id}, Item {item}")
            continue
            
        product_name = item_df['Product'].iloc[0]
        
        # Prepare time series data
        seq_length = 28  # Fixed sequence length
        ts_data = prepare_time_series_features(
            df[(df['Store_Id'] == store_id) & (df['Item'] == item)], 
            seq_length=seq_length
        )
        
        if ts_data is None:
            logger.warning(f"Could not prepare time series data for Store {store_id}, Item {item}")
            continue
        
        feature_dim = ts_data['feature_dim']
        scaler = ts_data['scalers'].get((store_id, item))
        
        if scaler is None:
            logger.warning(f"No scaler found for Store {store_id}, Item {item}")
            continue
        
        # Create future features
        try:
            future_data = create_future_features(
                df, store_id, item, last_date, 
                days_to_forecast, seq_length=seq_length
            )
        except Exception as e:
            logger.error(f"Error creating future features: {e}")
            continue
        
        if future_data is None:
            logger.warning(f"Could not create future features for Store {store_id}, Item {item}")
            continue
        
        # Try to load existing models - only LSTM model now as transformer is commented out
        lstm_model = load_model(store_id, item, feature_dim, model_type='lstm', device=device)
        # transformer_model = load_model(store_id, item, feature_dim, model_type='transformer', device=device)
        transformer_model = None  # Set to None since we're not using transformer model
        
        # Train models if not available - only LSTM now
        if lstm_model is None:
            logger.info(f"Training LSTM model for Store {store_id}, Item {item}")
            lstm_model = train_time_series_model(
                ts_data, model_type='lstm', epochs=30, batch_size=64, 
                learning_rate=0.001, device=device
            )
            save_model(lstm_model, store_id, item, model_type='lstm')
            save_scaler(scaler, store_id, item)
        
        # Transformer model training is removed
        # if transformer_model is None:
        #     logger.info(f"Training Transformer model for Store {store_id}, Item {item}")
        #     transformer_model = train_time_series_model(
        #         ts_data, model_type='transformer', epochs=30, batch_size=64, 
        #         learning_rate=0.001, device=device
        #     )
        #     save_model(transformer_model, store_id, item, model_type='transformer')
            
        # Add to model info - only showing LSTM model now
        model_info_data.append({
            'Store_Id': store_id,
            'Item': item,
            'Product': product_name,
            'LSTM_Model': f"model_{store_id}_{item}.pt",
            'Scaler': f"scaler_{store_id}_{item}.npz",
        })
        
        # Create model to use - only LSTM now
        models_to_use = []
        if lstm_model is not None:
            models_to_use.append(lstm_model)
        # Removed transformer model from ensemble
        # if transformer_model is not None:
        #     models_to_use.append(transformer_model)
            
        if not models_to_use:
            logger.error(f"No valid models available for Store {store_id}, Item {item}")
            continue
            
        # Only one model now (LSTM)
        weights = [1.0]  
            
        try:
            ensemble = ModelEnsemble(models_to_use, weights=weights)
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            continue
        
        # Make forecasts
        X_sequences = future_data['X_sequences']
        future_df = future_data['future_df']
        
        # Generate forecasts
        scaled_forecasts = ensemble.predict(X_sequences, device=device)
        
        # Inverse transform to get actual values
        try:
            if len(scaled_forecasts.shape) > 2:  # Need to reshape for scaler
                # Reshape to 2D for inverse_transform
                original_shape = scaled_forecasts.shape
                reshaped = scaled_forecasts.reshape(-1, 1)
                forecasts = scaler.inverse_transform(reshaped).reshape(original_shape)
            else:  # Already 2D or 1D
                forecasts = scaler.inverse_transform(scaled_forecasts)
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            # Fallback: use simple scaling instead
            mean = scaler.mean_[0] if hasattr(scaler, 'mean_') and len(scaler.mean_) > 0 else 0
            scale = scaler.scale_[0] if hasattr(scaler, 'scale_') and len(scaler.scale_) > 0 else 1
            forecasts = scaled_forecasts * scale + mean
        
        # Ensure non-negative values
        forecasts = np.maximum(0, forecasts)
        
        # Add forecasts to the future dataframe
        # Handle potential shape mismatch - ensure forecasts match the length of future_df
        forecast_values = forecasts.flatten()
        
        # Check for length mismatch and fix it
        if len(forecast_values) != len(future_df):
            logger.warning(f"Length mismatch: forecast values ({len(forecast_values)}) != future_df rows ({len(future_df)})")
            # Truncate or extend as needed
            if len(forecast_values) > len(future_df):
                # Truncate to match future_df length
                forecast_values = forecast_values[:len(future_df)]
            else:
                # Extend by repeating the last value
                padding = np.full(len(future_df) - len(forecast_values), forecast_values[-1])
                forecast_values = np.concatenate([forecast_values, padding])
            
            logger.info(f"Adjusted forecast length to match future_df length ({len(future_df)})")
        
        # Now assign the correctly sized forecast values
        future_df['Forecast'] = forecast_values
        
        # Add confidence intervals (simple approximation)
        # Uncertainty increases with days in future
        future_df['Std_Dev'] = future_df['Forecast'] * (0.1 + 0.005 * np.arange(1, days_to_forecast+1))
        future_df['Lower_Bound'] = np.maximum(0, future_df['Forecast'] - 1.96 * future_df['Std_Dev'])
        future_df['Upper_Bound'] = future_df['Forecast'] + 1.96 * future_df['Std_Dev']
        
        # Add to all forecasts
        all_forecasts.append(future_df)
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        # Get historical data for plotting
        hist_data = df[(df['Store_Id'] == store_id) & (df['Item'] == item)].sort_values('Date')
        hist_data = hist_data.iloc[-90:]  # Last 90 days
        
        # Plot historical data
        plt.plot(hist_data['Date'], hist_data['Sales'], 'b-', label='Historical Sales')
        
        # Plot forecast
        plt.plot(future_df['Date'], future_df['Forecast'], 'r-', label='Forecast')
        plt.fill_between(
            future_df['Date'],
            future_df['Lower_Bound'],
            future_df['Upper_Bound'],
            color='r', alpha=0.2, label='95% Confidence Interval'
        )
        
        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Sales Forecast for {product_name} (Store {store_id}, Item {item})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_dir = os.path.join(STATIC_DIR, 'images', 'time_series')
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.savefig(os.path.join(plot_dir, f'pred_{store_id}_{item}.png'), bbox_inches='tight')
        plt.close()
    
    # Combine all forecasts
    if not all_forecasts:
        logger.error("No forecasts were generated")
        return None
        
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    # Add metadata
    combined_forecasts['Forecast_Generated'] = datetime.now()
    combined_forecasts['Days_In_Future'] = combined_forecasts.groupby(['Store_Id', 'Item'])['Date'].rank().astype(int)
    
    # Save model info
    model_info_df = pd.DataFrame(model_info_data)
    model_info_df.to_csv(os.path.join(MODELS_DIR, 'time_series', 'model_info.csv'), index=False)
    
    logger.info(f"Generated forecasts for {len(store_items)} products over {days_to_forecast} days")
    return combined_forecasts


def save_forecasts(forecasts, output_file=PYTORCH_FORECASTS_FILE):
    """
    Save forecasts to CSV
    
    Args:
        forecasts: DataFrame with forecasts
        output_file: Path to save the CSV
    """
    logger.info(f"Saving forecasts to {output_file}")
    try:
        forecasts.to_csv(output_file, index=False)
        logger.info(f"Saved forecasts to {output_file}")
    except Exception as e:
        logger.error(f"Error saving forecasts: {e}")


def run_time_series_forecasting(data_file=COMBINED_DATA_FILE, days_to_forecast=30):
    """
    Main function to run the time series forecasting process
    
    Args:
        data_file: Path to the data file
        days_to_forecast: Number of days to forecast
        
    Returns:
        DataFrame with forecasts
    """
    logger.info(f"Starting time series forecasting process")
    
    try:
        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Loaded data with {len(df)} rows from {data_file}")
        
        # Generate forecasts
        forecasts = forecast_time_series(df, days_to_forecast=days_to_forecast)
        
        # Save forecasts
        if forecasts is not None:
            save_forecasts(forecasts)
            return forecasts
        else:
            logger.error("Failed to generate forecasts")
            return None
    
    except Exception as e:
        logger.error(f"Error in time series forecasting process: {e}", exc_info=True)
        raise


def main():
    """
    Main function to run when script is called directly
    """
    parser = argparse.ArgumentParser(description='Time series model training and forecasting')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--data-file', type=str, default=COMBINED_DATA_FILE, help='Path to input data file')
    args = parser.parse_args()
    
    run_time_series_forecasting(
        data_file=args.data_file,
        days_to_forecast=args.days
    )
    
    logger.info("Time series forecasting completed successfully")


if __name__ == "__main__":
    main()
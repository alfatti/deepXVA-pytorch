# TensorFlow to PyTorch Migration Guide

## Overview

This document details the conversion process from TensorFlow 2.x to PyTorch for the Deep xVA Solver.

## Major Changes Summary

### 1. Framework Imports

**TensorFlow:**
```python
import tensorflow as tf
from tensorflow import keras
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
```

### 2. Model Definition

**TensorFlow:**
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(64)
    
    def call(self, x, training=False):
        return self.dense(x)
```

**PyTorch:**
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = nn.Linear(input_dim, 64)
    
    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()
        return self.dense(x)
```

### 3. Trainable Variables

**TensorFlow:**
```python
self.y_init = tf.Variable(initial_value, dtype=tf.float64)
```

**PyTorch:**
```python
self.y_init = nn.Parameter(torch.tensor(initial_value, dtype=torch.float64))
```

### 4. Batch Normalization

**TensorFlow:**
```python
tf.keras.layers.BatchNormalization(
    momentum=0.99,
    epsilon=1e-6
)
```

**PyTorch:**
```python
nn.BatchNorm1d(
    num_features,
    momentum=0.01,  # Note: PyTorch uses 1 - TF momentum
    eps=1e-6
)
```

**Important:** PyTorch's momentum parameter is defined as `1 - tensorflow_momentum`.

### 5. Activation Functions

**TensorFlow:**
```python
tf.nn.relu(x)
tf.nn.sigmoid(x)
```

**PyTorch:**
```python
torch.relu(x)  # or F.relu(x)
torch.sigmoid(x)  # or F.sigmoid(x)
```

### 6. Operations

| Operation | TensorFlow | PyTorch |
|-----------|-----------|---------|
| Sum | `tf.reduce_sum(x, axis=1, keepdims=True)` | `torch.sum(x, dim=1, keepdim=True)` |
| Mean | `tf.reduce_mean(x, axis=0)` | `torch.mean(x, dim=0)` |
| Maximum | `tf.maximum(x, y)` | `torch.maximum(x, y)` |
| Minimum | `tf.minimum(x, y)` | `torch.minimum(x, y)` |
| Square | `tf.square(x)` | `torch.square(x)` |
| Abs | `tf.abs(x)` | `torch.abs(x)` |
| Matmul | `tf.matmul(x, y)` | `torch.matmul(x, y)` |
| Where | `tf.where(condition, x, y)` | `torch.where(condition, x, y)` |
| Stack | `tf.stack(tensors)` | `torch.stack(tensors)` |
| Concat | `tf.concat([x, y], axis=1)` | `torch.cat([x, y], dim=1)` |

### 7. Training Loop

**TensorFlow:**
```python
@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss = loss_fn(data)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**PyTorch:**
```python
def train_step(data):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(data)
    loss.backward()
    optimizer.step()
```

### 8. Learning Rate Schedule

**TensorFlow:**
```python
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[1000, 2000],
    values=[1e-2, 1e-3, 1e-4]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

**PyTorch:**
```python
# Manual implementation
def get_lr(step):
    boundaries = [1000, 2000]
    values = [1e-2, 1e-3, 1e-4]
    for i, boundary in enumerate(boundaries):
        if step < boundary:
            return values[i]
    return values[-1]

# In training loop
for step in range(num_steps):
    current_lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
```

Or use PyTorch schedulers:
```python
from torch.optim.lr_scheduler import MultiStepLR
scheduler = MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.1)
```

### 9. Data Type Management

**TensorFlow:**
```python
tf.keras.backend.set_floatx('float64')
tensor = tf.constant(value, dtype=tf.float64)
```

**PyTorch:**
```python
torch.set_default_dtype(torch.float64)
tensor = torch.tensor(value, dtype=torch.float64)
```

### 10. Device Management

**TensorFlow:**
```python
with tf.device('/GPU:0'):
    # operations
```

**PyTorch:**
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### 11. Saving and Loading Models

**TensorFlow:**
```python
model.save('model.h5')
model = tf.keras.models.load_model('model.h5')
```

**PyTorch:**
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 12. Converting NumPy Arrays

**TensorFlow:**
```python
# Automatic conversion in most cases
output = model(numpy_array)
```

**PyTorch:**
```python
# Explicit conversion needed
tensor = torch.tensor(numpy_array, dtype=torch.float64)
output = model(tensor)
result = output.detach().cpu().numpy()
```

### 13. Gradient Clipping

**TensorFlow:**
```python
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

**PyTorch:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 14. Custom Layers

**TensorFlow:**
```python
class CustomLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 10))
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w)
```

**PyTorch:**
```python
class CustomLayer(nn.Module):
    def __init__(self, input_dim):
        super(CustomLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(input_dim, 10))
    
    def forward(self, inputs):
        return torch.matmul(inputs, self.w)
```

## Specific Changes in Deep xVA Solver

### 1. BSDE Generator and Terminal Functions

Changed method names from `f_tf` and `g_tf` to `f_torch` and `g_torch` for clarity.

### 2. Subnet Architecture

**Key change in BatchNorm momentum:**
- TensorFlow uses momentum=0.99 (keeps 99% of running stats)
- PyTorch equivalent is momentum=0.01 (updates 1% per batch)

### 3. TensorArray Replacement

**TensorFlow:**
```python
history = tf.TensorArray(dtype, size=n)
history = history.write(i, value)
history = history.stack()
```

**PyTorch:**
```python
history = []
history.append(value)
history = torch.cat(history, dim=-1)
```

### 4. Training Mode Management

PyTorch requires explicit setting of training mode:
```python
model.train()  # Enable training mode
model.eval()   # Disable training mode (for inference)
```

## Common Pitfalls

### 1. In-place Operations

PyTorch doesn't allow in-place modifications of tensors that require gradients:
```python
# Wrong
x += y  

# Correct
x = x + y
```

### 2. Device Mismatch

All tensors in an operation must be on the same device:
```python
# Ensure both are on the same device
x = x.to(device)
y = y.to(device)
result = x + y
```

### 3. Gradient Accumulation

PyTorch accumulates gradients by default:
```python
optimizer.zero_grad()  # Always call before backward()
loss.backward()
optimizer.step()
```

### 4. NumPy Conversion

Convert to CPU before converting to NumPy:
```python
# Wrong
result = tensor.numpy()

# Correct
result = tensor.detach().cpu().numpy()
```

### 5. Model Evaluation

Always set model to eval mode for inference:
```python
model.eval()
with torch.no_grad():
    output = model(input)
```

## Performance Tips

1. **Use DataLoader** for efficient batching (not implemented in this version)
2. **Enable cudnn.benchmark** for CNN-heavy models:
   ```python
   torch.backends.cudnn.benchmark = True
   ```
3. **Use mixed precision** for faster training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```
4. **Compile models** (PyTorch 2.0+):
   ```python
   model = torch.compile(model)
   ```

## Testing the Conversion

Run all example scripts and verify:
1. Training convergence (loss curves)
2. Final Y0 values match TensorFlow version
3. Simulation outputs are similar
4. Exact solution comparisons are consistent

## Troubleshooting

### Issue: Different numerical results

**Cause:** Different random initialization or batch norm momentum

**Solution:** 
- Set random seeds: `torch.manual_seed(42)`
- Verify batch norm parameters
- Check activation functions

### Issue: Out of memory errors

**Cause:** PyTorch keeps computation graph by default

**Solution:**
```python
with torch.no_grad():
    # inference code
```

### Issue: Slow training

**Cause:** Not using GPU or inefficient tensor operations

**Solution:**
- Verify GPU usage: `print(next(model.parameters()).device)`
- Profile code to find bottlenecks
- Use PyTorch profiler

## Conclusion

The PyTorch conversion maintains the same mathematical algorithms while adapting to PyTorch's design patterns. The key differences are in training loop structure, device management, and some API naming conventions.

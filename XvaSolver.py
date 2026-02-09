import logging
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

DELTA_CLIP = 50.0


class XvaSolver(object):
    """The fully connected neural network model for XVA computation."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
       
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set dtype
        if self.net_config.dtype == "float64":
            self.dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            self.dtype = torch.float32
            torch.set_default_dtype(torch.float32)
        
        self.model = NonsharedModel(config, bsde).to(self.device)

        # Setup learning rate schedule
        try:
            lr_schedule = config.net_config.lr_schedule
        except AttributeError:
            self.lr_values = self.net_config.lr_values
            self.lr_boundaries = self.net_config.lr_boundaries
            lr_schedule = None
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.net_config.lr_values[0], eps=1e-8)

    def get_lr(self, step):
        """Get learning rate for current step."""
        for i, boundary in enumerate(self.lr_boundaries):
            if step < boundary:
                return self.lr_values[i]
        return self.lr_values[-1]

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in tqdm(range(self.net_config.num_iterations + 1)):
            # Update learning rate
            current_lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).item()
                y_init = self.model.y_init.detach().cpu().numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    print("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            
            self.train_step(self.bsde.sample(self.net_config.batch_size))
        
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x, v_clean, coll = inputs
        
        # Convert to tensors
        dw_t = torch.tensor(dw, dtype=self.dtype, device=self.device)
        x_t = torch.tensor(x, dtype=self.dtype, device=self.device)
        v_clean_t = torch.tensor(v_clean, dtype=self.dtype, device=self.device)
        coll_t = torch.tensor(coll, dtype=self.dtype, device=self.device)
        
        y_terminal = self.model((dw_t, x_t, v_clean_t, coll_t), training)
        g_terminal = self.bsde.g_torch(self.bsde.total_time, x_t[:, :, -1], v_clean_t[:, :, -1], coll_t[:, :, -1])
        
        delta = y_terminal - g_terminal
        
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP,
                                      torch.square(delta),
                                      2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))

        return loss

    def train_step(self, train_data):
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.loss_fn(train_data, training=True)
        loss.backward()
        self.optimizer.step()


class NonsharedModel(nn.Module):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde       
        self.dim = bsde.dim
        
        # Set dtype
        if self.net_config.dtype == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        
        # Initialize y_init and z_init as parameters
        self.y_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=self.net_config.y_init_range[0],
                                high=self.net_config.y_init_range[1],
                                size=[1]),
                dtype=self.dtype
            )
        )
        
        self.z_init = nn.Parameter(
            torch.tensor(
                np.random.uniform(low=-.1, high=.1, size=[1, self.eqn_config.dim]),
                dtype=self.dtype
            )
        )
        
        # Create subnet list
        self.subnet = nn.ModuleList([
            FeedForwardSubNet(config, bsde.dim) 
            for _ in range(self.bsde.num_time_interval - 1)
        ])
       
    def forward(self, inputs, training):
        dw, x, v_clean, coll = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        
        batch_size = dw.shape[0]
        all_one_vec = torch.ones(batch_size, 1, dtype=self.dtype, device=dw.device)
        
        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, self.z_init)
        
        for t in range(0, self.bsde.num_time_interval - 1):
            f_val = self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z, v_clean[:, :, t], coll[:, :, t])
            y = y - self.bsde.delta_t * f_val + torch.sum(z * dw[:, :, t], 1, keepdim=True)
            
            z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
        
        # terminal time
        f_val = self.bsde.f_torch(time_stamp[-1], x[:, :, -2], y, z, v_clean[:, :, -2], coll[:, :, -2])
        y = y - self.bsde.delta_t * f_val + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
        
        return y

    def predict_step(self, data):
        self.eval()
        with torch.no_grad():
            dw, x, v_clean, coll = data[0]
            time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
            
            batch_size = dw.shape[0]
            all_one_vec = torch.ones(batch_size, 1, dtype=self.dtype, device=dw.device)
            
            y = all_one_vec * self.y_init
            z = torch.matmul(all_one_vec, self.z_init)
            
            history = [y.unsqueeze(-1)]
            
            for t in range(0, self.bsde.num_time_interval - 1):
                f_val = self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z, v_clean[:, :, t], coll[:, :, t])
                y = y - self.bsde.delta_t * f_val + torch.sum(z * dw[:, :, t], 1, keepdim=True)
                
                history.append(y.unsqueeze(-1))
                z = self.subnet[t](x[:, :, t + 1], False) / self.bsde.dim
            
            # terminal time
            f_val = self.bsde.f_torch(time_stamp[-1], x[:, :, -2], y, z, v_clean[:, :, -2], coll[:, :, -2])
            y = y - self.bsde.delta_t * f_val + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
            
            history.append(y.unsqueeze(-1))
            history = torch.cat(history, dim=-1)
            
        return dw, x, v_clean, coll, history

    def simulate_path(self, sample_data):
        """Simulate path from numpy samples."""
        dw, x, v_clean, coll = sample_data
        
        # Convert to tensors
        dw_t = torch.tensor(dw, dtype=self.dtype, device=self.y_init.device)
        x_t = torch.tensor(x, dtype=self.dtype, device=self.y_init.device)
        v_clean_t = torch.tensor(v_clean, dtype=self.dtype, device=self.y_init.device)
        coll_t = torch.tensor(coll, dtype=self.dtype, device=self.y_init.device)
        
        _, _, _, _, history = self.predict_step([(dw_t, x_t, v_clean_t, coll_t)])
        
        return history.cpu().numpy()


class FeedForwardSubNet(nn.Module):
    def __init__(self, config, dim):
        super(FeedForwardSubNet, self).__init__()
        
        num_hiddens = config.net_config.num_hiddens
        
        # Set dtype
        if config.net_config.dtype == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(dim if i == 0 else num_hiddens[i-1] if i <= len(num_hiddens) else dim,
                          momentum=0.01,  # PyTorch uses 1-momentum of TensorFlow
                          eps=1e-6)
            for i in range(len(num_hiddens) + 2)
        ])
        
        # Dense layers
        self.dense_layers = nn.ModuleList([
            nn.Linear(dim if i == 0 else num_hiddens[i-1], num_hiddens[i], bias=False)
            for i in range(len(num_hiddens))
        ])
        
        # Final output layer
        input_dim = num_hiddens[-1] if num_hiddens else dim
        self.dense_layers.append(nn.Linear(input_dim, dim, bias=True))

    def forward(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense"""
        
        # Set training mode
        if training:
            self.train()
        else:
            self.eval()
        
        x = self.bn_layers[0](x)
        
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x)
            x = torch.relu(x)
        
        x = self.dense_layers[-1](x)
        
        return x

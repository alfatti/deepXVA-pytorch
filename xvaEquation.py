from equation import Equation
import numpy as np
import torch
from scipy.stats import multivariate_normal as normal

class PricingForward(Equation):
    def __init__(self, eqn_config):
        super(PricingForward, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r        
        self.useExplict = False  # whether to use explicit formula to evaluate dynamics of x

    def sample(self, num_sample):      
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim == 1:
            dw_sample = np.expand_dims(dw_sample, axis=0)
            dw_sample = np.swapaxes(dw_sample, 0, 1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict:  # use analytic solution of linear SDE
            factor = np.exp((self.r - (self.sigma**2) / 2) * self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   # use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
        
        return dw_sample, x_sample   

    def f_torch(self, t, x, y, z):
        return -self.r * y
   
    def g_torch(self, t, x):
        return x - self.strike
    

class BasketOption(Equation):
    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma    # volatility 
        self.rate = eqn_config.r    # risk-free rate 
        self.useExplict = True  # whether to use explicit formula to evaluate dynamics of x

    def sample(self, num_sample, use_gpu=False):
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim == 1:
            dw_sample = np.expand_dims(dw_sample, axis=0)
            dw_sample = np.swapaxes(dw_sample, 0, 1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict:  # use analytic solution of linear SDE
            factor = np.exp((self.rate - (self.sigma**2) / 2) * self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   # use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = (1 + self.rate * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])       
        
        return dw_sample, x_sample 

    def f_torch(self, t, x, y, z):
        return -self.rate * y

    def g_torch(self, t, x):
        temp = torch.sum(x, 1, keepdim=True)
        return torch.maximum(temp - self.strike, torch.zeros_like(temp))


class CallOption(Equation):
    def __init__(self, eqn_config):
        super(CallOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = False  # whether to use explicit formula to evaluate dynamics of x

    def sample(self, num_sample):      
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim == 1:
            dw_sample = np.expand_dims(dw_sample, axis=0)
            dw_sample = np.swapaxes(dw_sample, 0, 1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict:  # use analytic solution of linear SDE
            factor = np.exp((self.r - (self.sigma**2) / 2) * self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   # use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
        
        return dw_sample, x_sample  

    def f_torch(self, t, x, y, z):
        return -self.r * y

    def g_torch(self, t, x):
        return torch.maximum(x - self.strike, torch.zeros_like(x))

    def SolExact(self, t, x):
        d1 = (np.log(x / self.strike) + (self.r + 0.5 * self.sigma ** 2) * (self.total_time - t)) / (self.sigma * np.sqrt(self.total_time - t))
        d2 = (np.log(x / self.strike) + (self.r - 0.5 * self.sigma ** 2) * (self.total_time - t)) / (self.sigma * np.sqrt(self.total_time - t))

        call = (x * normal.cdf(d1, 0.0, 1.0) - self.strike * np.exp(-self.r * (self.total_time - t)) * normal.cdf(d2, 0.0, 1.0))
        return call
    

class CVA(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(CVA, self).__init__(dim, total_time, num_time_interval)
        self._x0 = 100
        self._x_init = np.ones(self._dim) * self._x0  # x0 dynamics forward
        self._sigma = 0.25    # volatility dynamics forward
        self._r = 0.0        # interest rate
        self._recoveryC = 0.3
        self._lambdaC = 0.10
        self._recoveryB = 0.4
        self._lambdaB = 0.01

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,     
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t

        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init       

        factor = np.exp((self._r - (self._sigma**2) / 2) * self._delta_t)
        for i in range(self._num_time_interval):    
            x_sample[:, :, i + 1] = (factor * np.exp(self._sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample   
   
    def f_torch(self, t, x, y, z, vHat):
        return (-(1.0 - self._recoveryC) * torch.maximum(-vHat, torch.zeros_like(vHat)) * self._lambdaC - 
                (self._r + self._lambdaC + self._lambdaB) * y + 
                (1.0 - self._recoveryB) * torch.maximum(vHat, torch.zeros_like(vHat)) * self._lambdaB - 
                (self._r + self._lambdaC + self._lambdaB) * y)

    def g_torch(self, t, x):
        return 0.0


class Portfolio():   
    # takes in a list of NonsharedModel (financial derivatives) and can be passed as a clean_value process to an XVA object
    def __init__(self, assets, weights=None):
        # assets: a list of NonsharedModel
        # !!!Warning!!! Assumes independence between random drivers of each asset    
        self.assets = assets
        self.num_assets = len(assets)
        if weights is None:
            self.weights = [1 for i in range(self.num_assets)]
        else:
            self.weights = weights
        self.dim = sum([asset.bsde.dim for asset in assets])

    def sample(self, num_sample):
        w, x, v = self.assets[0].predict(self.assets[0].bsde.sample(num_sample))
        if self.num_assets > 1:            
            v = v * self.weights[0]
            for i, weight in enumerate(self.weights[1:]):
                w_, x_, v_ = self.assets[i+1].sample(self.assets[0].bsde.sample(num_sample))
                w = np.concatenate((w, w_), axis=1)
                x = np.concatenate((x, x_), axis=1)
                v = v + v_ * weight
        return w, x, v
    

class XVA(Equation):
    def __init__(self, eqn_config, clean_value):
        self.dim = clean_value.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None       
        self.rate = eqn_config.r  # risk-free return
        self.intensityB = eqn_config.intensityB  # default intensity of the bank
        self.intensityC = eqn_config.intensityC  # default intensity of the counterparty
        self.r_fl = eqn_config.r_fl  # unsecured funding lending rate
        self.r_fb = eqn_config.r_fb  # unsecured funding borrowing rate
        self.r_cl = eqn_config.r_cl  # interest rate on posted collateral
        self.r_cb = eqn_config.r_cb  # interest rate on received collateral
        self.clean_value = clean_value  # Class NonsharedModel, representing the clean value process
        self.collateral = eqn_config.collateral
        self.isXVA = True  # indicating that we are dealing with xva
        self.dim = clean_value.dim
        try:
            self.num_assets = clean_value.num_assets  # when clean_value is of class Portfolio
        except AttributeError:
            self.num_assets = None  # when clean_value is of class NonsharedModel, set to None

        # setting (default) value for recovery rate
        try:
            self.R_B = eqn_config.Recovery_Bank
        except AttributeError:
            self.R_B = 0.6
        try:
            self.R_C = eqn_config.Recovery_Counterparty
        except AttributeError:
            self.R_C = 0.6

    def sample(self, num_sample):
        if self.num_assets is None:
            return self.clean_value.predict(self.clean_value.bsde.sample(num_sample))
        else:
            return self.clean_value.sample(num_sample)

    def f_torch(self, t, x, y, z, v_clean):
        # x: underlying, y: xva, z: control
        fva = ((self.r_fl - self.rate) * torch.maximum(v_clean - y - self.collateral, torch.zeros_like(v_clean)) - 
               (self.r_fb - self.rate) * torch.maximum(self.collateral + y - v_clean, torch.zeros_like(v_clean)))
        discount = -(self.rate + self.intensityB + self.intensityC) * y

        return fva + discount
    
    def g_torch(self, t, x, v_clean):
        return 0.

    def monte_carlo(self, num_sample=1024):  # monte carlo estimation of CVA and DVA
        from tqdm import tqdm
        
        print('CVA and DVA Monte Carlo estimation started')
        discount = np.exp(-(self.rate + self.intensityB + self.intensityC) * np.linspace(0, self.total_time, self.num_time_interval + 1))
       
        estimate = []           
        for i in tqdm(range(num_sample // 1024)):  # split into batches to estimate
            v = self.sample(1024)[1]  # shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1 - self.R_C) * discount * np.maximum(self.collateral - v, 0) * self.intensityC
            phi_dva = (1 - self.R_B) * discount * np.maximum(v - self.collateral, 0) * self.intensityB

            # trapezoidal rule
            cva = np.sum(phi_cva, axis=-1) - (phi_cva[:, :, -1] + phi_cva[:, :, 0]) / 2
            dva = np.sum(phi_dva, axis=-1) - (phi_dva[:, :, -1] + phi_dva[:, :, 0]) / 2

            estimate += list(dva[:, 0] - cva[:, 0])

        if num_sample % 1024 != 0:  # calculate the remaining number of samples
            v = self.sample(num_sample % 1024)[1]  # shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1 - self.R_C) * discount * np.maximum(self.collateral - v, 0) * self.intensityC
            phi_dva = (1 - self.R_B) * discount * np.maximum(v - self.collateral, 0) * self.intensityB

            # trapezoidal rule
            cva = np.sum(phi_cva, axis=-1) - (phi_cva[:, :, -1] + phi_cva[:, :, 0]) / 2
            dva = np.sum(phi_dva, axis=-1) - (phi_dva[:, :, -1] + phi_dva[:, :, 0]) / 2

            estimate += list(dva[:, 0] - cva[:, 0])
        
        estimate = np.array(estimate) * self.delta_t  # times time-interval (height of a trapezium)
        mean = np.mean(estimate)
        std = np.std(estimate) / np.sqrt(num_sample)

        return mean, [mean - 3 * std, mean + 3 * std]

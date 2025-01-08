import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random 

class BaseSeries:
    def __init__(self):
        self.data = None
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        if not isinstance(other, DataSeries):
            raise ValueError('Can only add DataSeries objects')
        
        if len(self.index) != len(other.index):
            raise ValueError('Indexes must be equal')
        
        return DataSeries(data=self.data + other.data, index=self.index)
    
    def __sub__(self, other):
        if not isinstance(other, DataSeries):
            raise ValueError('Can only subtract DataSeries objects')
        
        if len(self.index) != len(other.index):
            raise ValueError('Indexes must be equal')
        
        return DataSeries(data=self.data - other.data, index=self.index)
    
    def __mul__(self, other):
        if not isinstance(other, DataSeries):
            raise ValueError('Can only multiply DataSeries objects')
        
        if len(self.index) != len(other.index):
            raise ValueError('Indexes must be equal')
        
        return DataSeries(data=self.data * other.data, index=self.index)
    
    def __truediv__(self, other):
        if not isinstance(other, DataSeries):
            raise ValueError('Can only divide DataSeries objects')
        
        if len(self.index) != len(other.index):
            raise ValueError('Indexes must be equal')
        
        return DataSeries(data=self.data / other.data, index=self.index)
    
    def __pow__(self, other):
        if not isinstance(other, DataSeries):
            raise ValueError('Can only exponentiate DataSeries objects')
        
        if len(self.index) != len(other.index):
            raise ValueError('Indexes must be equal')
        
        return DataSeries(data=self.data ** other.data, index=self.index)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self / other
    
    def __rpow__(self, other):
        return self ** other
    
    def __repr__(self):
        if not hasattr(self, 'display_data'):
            self.display_data = ", ".join([f"{d:.2f}" for d in self.data[:5]])[:-2]
        return f'{self.__class__.__name__}([{self.display_data}...], length={len(self)})'

class DataSeries(BaseSeries):
    def __init__(self, path: str | None = None, data: np.ndarray | None = None, index: np.ndarray | None = None, denormalize_params: tuple | None = None):
        if path is not None:
            self.path = path
            df = pd.read_excel(path, index_col=0)
            self.data = df['y'].to_numpy()
            self.index = df.index.astype(int).to_numpy()
        elif data is not None:
            self.data = data
            self.index = np.linspace(0, 1, len(data), dtype=float) if index is None else index
            self.path = None
        else:
            raise ValueError('Either path or data must be provided')
        
        self.denormalize_params = denormalize_params
        
    def normalize(self):
        denormalize_params = (self.data.mean(), self.data.std(), self.index.max(), self.index.min())
        data = (self.data - self.data.mean()) / self.data.std()
        index = (self.index - self.index.min()) / (self.index.max() - self.index.min())
        return DataSeries(data=data, index=index, denormalize_params=denormalize_params)
    
    def denormalize(self):
        if self.denormalize_params is None:
            raise ValueError('Data is not normalized')
        
        data = self.data * self.denormalize_params[1] + self.denormalize_params[0]
        index = self.index * (self.denormalize_params[2] - self.denormalize_params[3]) + self.denormalize_params[3]
        return DataSeries(data=data, index=index.astype(int))
    
    def _denormalize_number(self, number):
        return number * self.denormalize_params[1] + self.denormalize_params[0]
    
    def remove_outliers(self, global_threshold=2, local_threshold=3, local_lookback=0.05):
        self._remove_outliers_global(global_threshold)
        self._remove_outliers_continuos(local_threshold, local_lookback)
        return self

    def _remove_outliers_global(self, threshold=3):
        z = np.abs((self.data - self.data.mean()) / self.data.std())
        self.data = self.data[z < threshold]
        self.index = self.index[z < threshold]
    
    def _remove_outliers_continuos(self, threshold=3, lookback=0.1):
        lookback = int(len(self.data) * lookback)
        index_format = self.index.dtype
        self.index = self.index.astype(float)
        for i in range(lookback, len(self.data)):
            z = np.abs((self.data[i-lookback:i] - self.data[i-lookback:i].mean()) / (self.data[i-lookback:i].std() + 1e-6))
            self.data[i-lookback:i] = np.where(z > threshold, np.nan, self.data[i-lookback:i])
            self.index[i-lookback:i] = np.where(z > threshold, np.nan, self.index[i-lookback:i])
        self.data = self.data[~np.isnan(self.data)]
        self.index = self.index[~np.isnan(self.index)].astype(index_format)

    def plot(self, mode=None, **kwargs):
        plt.plot(self.index, self.data, **kwargs)
        if mode != 'add':
            plt.grid(True)
            plt.show()
    
    def denoise_ls(self, H: np.ndarray):
        if isinstance(H, tuple):
            denoise_func = H[1]
            H = H[0]
            denoised, params = denoise_func(H)
            return DataSeries(data=denoised, index=self.index, denormalize_params=self.denormalize_params), params

        params = np.linalg.pinv(H) @ self.data
        denoised = H @ params
        return DataSeries(data=denoised, index=self.index, denormalize_params=self.denormalize_params), params
    
    def denoise_gd(self, H: np.ndarray, learning_rate=1e-5, max_iter=int(1e7), lambda_reg=0.1, early_stopping='grad'):
        if isinstance(H, tuple):
            H = H[0]

        assert early_stopping in ['grad', 'loss', False], 'Invalid early stopping criterion'
        
        params = np.random.randn(H.shape[1])
        cached_loss = 0

        for _ in range(max_iter):
            prediction = H @ params
            
            loss = np.mean((self.data - prediction) ** 2) + lambda_reg * np.sum(params ** 2)
            
            gradient = -2 * H.T @ (self.data - prediction) + 2 * lambda_reg * params
            
            params -= learning_rate * gradient
            
            if (early_stopping == 'loss' and np.abs(cached_loss - loss) < 1e-6) or \
            (early_stopping == 'grad' and np.sum(gradient ** 2) < 1):
                break
            
            cached_loss = loss

        denoised = H @ params
        return DataSeries(data=denoised, index=self.index, denormalize_params=self.denormalize_params), params
    
    def sample(self, factor):
        perm = random.sample(range(len(self.data)), int(len(self.data) * factor))
        indices = np.sort(perm)
        self.data = self.data[indices]
        self.index = self.index[indices]
        return self
    
    def copy(self):
        return DataSeries(data=self.data.copy(), index=self.index.copy(), denormalize_params=self.denormalize_params)

def H(data, kind='ramp', **kwargs):
    if 'ramp' in kind.lower():
        H = np.vander(data.index, 2)
        return H
    elif 'sin' in kind.lower():
        f = kwargs.get('f')
        if f is None:
            raise ValueError('f must be provided for "sin" kind')
        
        mult = 2 * np.pi * f
        H = np.column_stack((np.cos(mult * data.index), np.sin(mult * data.index)))
        
        def sin_denoise(H):
            params = np.linalg.pinv(H) @ data.data

            A = np.sqrt(np.sum(params**2))
            phi = np.arctan2(-params[1], params[0])
            # print(f'Amplitude: {A}, Phase: {phi:.2}rad')
            denoised = A * np.cos(mult * data.index - phi * np.pi / 180)

            return denoised, np.array([A, phi])
        
        return H, sin_denoise
    elif 'dc' in kind.lower():
        return np.ones((len(data), 1))
    else:
        lambda_reg = kwargs.get('lambda_reg')
        if lambda_reg is None:
            raise ValueError('lambda_reg must be provided for "piecewise" kind')
        H_piecewise = np.diff(np.eye(len(data)), axis=0)
        def piecewise_denoise(H):
            denoised = np.linalg.inv(np.eye(len(data)) + lambda_reg * H @ H.T) @ data.data
            return denoised, None
        
        return H_piecewise.T, piecewise_denoise
    
def plot_multiple(series_list, titles=None, labels=None, suptitle=None, dark_mode=True):
    series_list = list(series_list)
    
    plt.style.use('bmh')

    plt.figure(figsize=(18, 3))
    if suptitle is not None:
        plt.suptitle(suptitle, y=1.1)

    for i, series in enumerate(series_list):
        ax = plt.subplot(1, len(series_list), i + 1)

        if isinstance(series, tuple):
            for l_idx, serie in enumerate(series):
                serie.plot('add', label=labels[l_idx] if labels is not None else None)
                if labels:
                    plt.legend()
        else:
            series.plot('add')

        if titles is not None:
            titles = list(titles)
            plt.title(titles[i])
    
    plt.show()
    
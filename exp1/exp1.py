#%%
# Imports
from utils import DataSeries, H, plot_multiple

#%%
# Load data
datadict = {
    'DC 1': 'data/DC_noise1.xlsx', 
    'DC 2': 'data/DC_noise2.xlsx',
    'Ramp': 'data/ramp_noise.xlsx',
    'Sine': 'data/sinusoid_noise.xlsx',
    'Piecewise': 'data/noisy_signal.xlsx',
}

data_series = [DataSeries(path) for path in datadict.values()]

plot_multiple(data_series, datadict.keys(), suptitle='Original data')

#%%
# Remove outliers
cleaned = [ds.copy().remove_outliers() for ds in data_series.copy()]
plot_multiple(cleaned, ['Cleaned: ' + title for title in datadict.keys()], suptitle='Cleaned data')

#%%
# Normalize
normalized = [ds.copy().normalize() for ds in cleaned.copy()]
plot_multiple(normalized, ['Normalized: ' + title for title in datadict.keys()], suptitle='Normalized data')

#%%
# Least squares

l_kwargs = [{}, {} , {}, {'f': 5}, {'lambda_reg': 5}]

ls_denoised, ls_params = zip(*[ds.denoise_ls(H(ds, kind, **kwargs)) for ds, kind, kwargs in zip(normalized, datadict.values(), l_kwargs)])
plot_multiple(ls_denoised, ['Denoised (N): ' + title for title in datadict.keys()], suptitle='Denoised but Normalized (LS)')

#%%
# Denormalize
ls_denoised = [ds.denormalize() for ds in ls_denoised]
plot_multiple(ls_denoised, ['Denoised: ' + title for title in datadict.keys()], suptitle='Denoised data (LS)')

#%%
# Compare with original
plot_multiple(zip(data_series, ls_denoised), ['Denoised: ' + title for title in datadict.keys()], suptitle='Comparison with original data')
# %%
# Gradient Descent
normalized = normalized[:-1]
cleaned = cleaned[:-1]
datadict = {k: v for k, v in datadict.items() if k != 'Piecewise'}
l_kwargs = [{}, {} , {}, {'f': 5}]

gd_denoised, gd_params = zip(*[ds.denoise_gd(H(ds, kind, **kwargs)) for ds, kind, kwargs in zip(normalized, datadict.values(), l_kwargs)])
plot_multiple(gd_denoised, ['Denoised (N): ' + title for title in datadict.keys()], suptitle='Denoised but Normalized (GD)')

# %%
gd_denoised = [ds.denormalize() for ds in gd_denoised]
plot_multiple(gd_denoised, ['Denoised: ' + title for title in datadict.keys()], suptitle='Denoised data (GD)')
# %%
plot_multiple(zip(data_series, gd_denoised), ['Denoised: ' + title for title in datadict.keys()], suptitle='Comparison with original data')
# %%
# All three
plot_multiple(zip(data_series, gd_denoised, ls_denoised), ['Denoised: ' + title for title in datadict.keys()], labels=['Original', 'GD', 'LS'], suptitle='Comparison with original data')
# %%
# Subsampled LS

subsample_frac = 0.1
sdata_series = [ds.copy().sample(subsample_frac) for ds in cleaned]

plot_multiple(sdata_series, datadict.keys(), suptitle='Subsampled data')
# %%
plot_multiple(zip(cleaned, sdata_series), datadict.keys(), ['Original', 'Subsampled'], suptitle='Comparison between original and subsampled data')
# %%
snormalized = [ds.normalize() for ds in sdata_series]
plot_multiple(snormalized, ['Normalized: ' + title for title in datadict.keys()], suptitle='Normalized data')
# %%
l_kwargs = [{}, {} , {}, {'f': 5}, {'lambda_reg': 5}]

sls_denoised, sls_params = zip(*[ds.denoise_ls(H(ds, kind, **kwargs)) for ds, kind, kwargs in zip(snormalized, datadict.values(), l_kwargs)])
plot_multiple(sls_denoised, ['Denoised (N): ' + title for title in datadict.keys()], suptitle='Denoised but Normalized (SLS)')
# %%
sls_denoised = [ds.denormalize() for ds in sls_denoised]
plot_multiple(sls_denoised, ['Denoised: ' + title for title in datadict.keys()], suptitle='Denoised data (SLS)')
# %%
plot_multiple(zip(data_series, sdata_series, sls_denoised), ['Denoised: ' + title for title in datadict.keys()], labels=['Original', 'Subsampled', 'LS'], suptitle='Comparison with original data')
# %%

datadict = {k: v for k, v in datadict.items() if k != 'Piecewise'}
l_kwargs = [{}, {} , {}, {'f': 5}]

sgd_denoised, sgd_params = zip(*[ds.denoise_gd(H(ds, kind, **kwargs)) for ds, kind, kwargs in zip(snormalized, datadict.values(), l_kwargs)])
plot_multiple(sgd_denoised, ['Denoised (N): ' + title for title in datadict.keys()], suptitle='Denoised but Normalized (SGD)')
# %%

sgd_denoised = [ds.denormalize() for ds in sgd_denoised]
plot_multiple(sgd_denoised, ['Denoised: ' + title for title in datadict.keys()], suptitle='Denoised data (SGD)')
# %%

plot_multiple(zip(data_series, sdata_series, sgd_denoised), ['Denoised: ' + title for title in datadict.keys()], labels=['Original', 'Subsampled', 'SGD'], suptitle='Comparison with original data')
# %%
print("Least Squares:\n", ls_params[:-1])
print("Gradient Descent:\n", gd_params)
print("Subsampled Least Squares:\n", sls_params[:-1])
print("Subsampled Gradient Descent:\n", sgd_params)
# %%
plot_multiple(zip(cleaned, gd_denoised, ls_denoised, sgd_denoised, sls_denoised), ['Denoised: ' + title for title in datadict.keys()], ['Original', 'GD', 'LS', 'SGD', 'SLS'], suptitle='All of Them')
# %%

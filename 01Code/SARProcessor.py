import netCDF4 as nc
import os
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import numpy as np

class SARProcessor:

    def __init__(self,src_path,file_name='',lat=0,lon=0,width=0.5,height=0.5):
        self.src_path = src_path
        self.file_name = file_name
        self.file_path = os.path.join(src_path,file_name)

        # Add target coordinates
        self.lat = lat
        self.lon = lon

        # Add the width and height of the analysis window
        self.width = width
        self.height = height

    ############## METHODS ############
    def print_info(self):
            """
            Get the variable names and their dimensions from a NetCDF file.
    
            Parameters:
            file_name : str
                Name of the NetCDF file to read.
    
            Returns:
            var_info : dict
                Dictionary containing variable names as keys and their dimensions as values.
            """
            print(f"Getting variable names and dimensions from NetCDF file: {self.file_path}")
            with nc.Dataset(f"{os.path.join(self.src_path, self.file_path)}", 'r') as dataset:
                var_info = {var: (dataset.variables[var].dimensions, dataset.variables[var].shape) for var in dataset.variables}
            print(var_info)

    def read_file(self):
        self.ds = xr.open_dataset(self.file_path)

        print(self.ds)

        return self.ds

    def haversine_distance(self, lat1, lon1, lat2, lon2, earth_radius_km=6371.0):
        """
        Calculate the Haversine (great-circle) distance between two points on Earth.

        Parameters:
        -----------
        lat1, lon1 : float or np.ndarray
            Latitude and longitude of the first point(s) in degrees.
        lat2, lon2 : float or np.ndarray
            Latitude and longitude of the second point(s) in degrees.
        earth_radius_km : float, optional
            Radius of the Earth in kilometers (default is 6371.0 km).
            Use 3958.8 for miles.

        Returns:
        --------
        float or np.ndarray
            Great-circle distance in kilometers.
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        # Differences in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arcsin(np.sqrt(a))

        return earth_radius_km * c

    def find_minimum_distance(self, ds, target_lat, target_lon):
        # Calculate squared distance to target for every grid point
        distance = self.haversine_distance(ds.lat, ds.lon, target_lat, target_lon)

        # Find the 2D indices of the minimum distance
        min_dist_idx = distance.argmin(dim=['y', 'x'])

        y_idx = int(min_dist_idx['y'])
        x_idx = int(min_dist_idx['x'])

        return x_idx, y_idx

    def find_coordinates_index(self, ds, target_lat, target_lon):
        x_max_len = ds.sizes['x']
        y_max_len = ds.sizes['y']

        # --- Step 1: Coarse Search (skip 100) ---
        skip_1 = 1000
        ds_step1 = ds.isel(x=slice(0, None, skip_1), y=slice(0, None, skip_1))
        x_idx, y_idx = self.find_minimum_distance(ds_step1, target_lat, target_lon)
        
        # Scale back to full dataset coordinates
        x_abs = x_idx * skip_1
        y_abs = y_idx * skip_1

        # Define bounded box around the coarse estimate
        x_min = max(0, x_abs - skip_1)
        x_max = min(x_max_len, x_abs + skip_1 + 1)
        y_min = max(0, y_abs - skip_1)
        y_max = min(y_max_len, y_abs + skip_1 + 1)

        # --- Step 2: Medium Search (skip 100) ---
        skip_2 = 100
        ds_step2 = ds.isel(x=slice(x_min, x_max, skip_2), y=slice(y_min, y_max, skip_2))
        x_idx, y_idx = self.find_minimum_distance(ds_step2, target_lat, target_lon)
        
        # Map back to full dataset coordinates
        x_abs = x_min + (x_idx * skip_2)
        y_abs = y_min + (y_idx * skip_2)

        # Define fine bounded box
        x_min = max(0, x_abs - skip_2)
        x_max = min(x_max_len, x_abs + skip_2 + 1)
        y_min = max(0, y_abs - skip_2)
        y_max = min(y_max_len, y_abs + skip_2 + 1)

        # --- Step 3: Fine Search (skip 1 / full resolution) ---
        ds_reduced = ds.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
        x_idx, y_idx = self.find_minimum_distance(ds_reduced, target_lat, target_lon)
        
        # Final global indices
        final_x = x_min + x_idx
        final_y = y_min + y_idx

        return final_x, final_y, ds_reduced

    def lee_filter_2d(self, arr, size=5, cu=None):
        """
        2D Lee Speckle Filter operating on a NumPy array.
        """
        # Handle NaN values safely by filling temporarily
        nan_mask = np.isnan(arr)
        arr_filled = np.nan_to_num(arr, nan=0.0)
        
        arr_filled = arr_filled.astype(np.float64)
        
        # Local statistics via scipy uniform filter
        img_mean = uniform_filter(arr_filled, size=size)
        img_sqr_mean = uniform_filter(arr_filled**2, size=size)
        img_var = np.maximum(0, img_sqr_mean - img_mean**2)

        # Noise coefficient estimation
        if cu is None:
            valid_vars = img_var[~nan_mask]
            valid_means = img_mean[~nan_mask]
            cu = np.sqrt(np.percentile(valid_vars / (valid_means**2 + 1e-8), 5))

        var_noise = (cu * img_mean)**2

        # Lee Weight calculation
        weight = img_var / (img_var + var_noise + 1e-8)
        weight = np.clip(weight, 0.0, 1.0)

        # Filtered array
        output = img_mean + weight * (arr_filled - img_mean)
        
        # Restore original NaN values (e.g., ocean/no-data masks)
        output[nan_mask] = np.nan
        
        return output

    def _cfar_mask_2d_numpy(self, arr, num_guard=3, num_ref=6, pfa=1e-4):
        """NumPy 2D CA-CFAR implementation operating on spatial arrays."""
        nan_mask = np.isnan(arr)
        arr_filled = np.nan_to_num(arr, nan=0.0).astype(np.float64)

        guard_size = 2 * num_guard + 1
        total_size = 2 * (num_guard + num_ref) + 1
        n_ref_cells = (total_size**2) - (guard_size**2)

        # Scipy uniform filter to compute local reference sums
        sum_total = uniform_filter(arr_filled, size=total_size) * (total_size**2)
        sum_guard = uniform_filter(arr_filled, size=guard_size) * (guard_size**2)

        # Calculate noise power and dynamic threshold factor (alpha)
        noise_power = (sum_total - sum_guard) / n_ref_cells
        alpha = n_ref_cells * (pfa**(-1.0 / n_ref_cells) - 1.0)
        threshold = alpha * noise_power

        # Target detection decision
        target_mask = (arr_filled > threshold) & (~nan_mask)

        arr_clean = arr.copy()
        arr_clean[target_mask] = np.nan

        return arr_clean, target_mask


    def xarray_cfar_masking(self, ds, var_name='Sigma0_VV', spatial_dims=None, 
                            num_guard=3, num_ref=6, pfa=1e-4):
        """
        Apply CFAR Target Masking directly to an xarray Dataset or DataArray.
        Auto-detects spatial dimensions if spatial_dims is None.
        """
        da = ds[var_name] if isinstance(ds, xr.Dataset) else ds

        # Auto-detect spatial dimensions in exact dataset order
        if spatial_dims is None:
            spatial_dims = da.dims[-2:]
        
        # Ensure spatial_dims match the exact order in the DataArray
        spatial_dims = tuple(d for d in da.dims if d in spatial_dims)

        def _ufunc_wrapper(arr):
            cleaned, mask = _self.cfar_mask_2d_numpy(
                arr, num_guard=num_guard, num_ref=num_ref, pfa=pfa
            )
            return cleaned, mask

        # Execute ufunc across x and y spatial dimensions
        da_clean, da_mask = xr.apply_ufunc(
            _ufunc_wrapper,
            da,
            input_core_dims=[list(spatial_dims)],
            output_core_dims=[list(spatial_dims), list(spatial_dims)],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[da.dtype, bool]
        )

        da_clean.name = f"{var_name}_no_targets"
        da_mask.name = "target_mask"

        if isinstance(ds, xr.Dataset):
            ds_out = ds.copy()
            ds_out[da_clean.name] = da_clean
            ds_out[da_mask.name] = da_mask
            return ds_out
        else:
            return xr.merge([da_clean, da_mask])

    def compute_xarray_welch2d(self, da, spatial_dims=('y', 'x'), tile_size=(256, 256), overlap=0.5, window='hanning', return_db=True):
        """
        Computes 2D Welch Power Spectral Density (PSD) on an xarray DataArray 
        and returns a DataArray with centered spatial frequency axes (cycles per unit).

        Parameters:
        -----------
        da : xarray.DataArray
            Input 2D SAR spatial DataArray.
        spatial_dims : tuple of str
            The 2D spatial dimension names, e.g., ('y', 'x') or ('lat', 'lon').
        tile_size : tuple of int (ty, tx)
            Size of the sliding sub-tiles in pixels (default 256x256).
        overlap : float (0.0 to 1.0)
            Fractional overlap between adjacent tiles (0.5 = 50% overlap).
        window : str
            Type of 2D windowing function ('hanning', 'hamming', or 'boxcar').
        return_db : bool
            If True, converts PSD to decibels: 10 * log10(PSD).

        Returns:
        --------
        xr.DataArray
            2D Welch PSD with centered frequency coordinates (freq_y, freq_x).
        """
        dim_y, dim_x = spatial_dims
        
        # 1. Fill NaNs with mean to prevent FFT corruption
        da_filled = da.fillna(da.mean().item())
        arr = da_filled.values.astype(np.float64)
        
        ny, nx = arr.shape
        ty, tx = tile_size

        # 2. Extract spatial resolution (dx, dy) from coordinate vectors
        dy = np.abs(np.diff(da[dim_y].values)[0])
        dx = np.abs(np.diff(da[dim_x].values)[0])

        # 3. Calculate step sizes based on overlap percentage
        step_y = max(1, int(ty * (1.0 - overlap)))
        step_x = max(1, int(tx * (1.0 - overlap)))

        # 4. Construct 2D tapering window and energy normalization factor
        if window == 'hanning':
            win_2d = np.outer(np.hanning(ty), np.hanning(tx))
        elif window == 'hamming':
            win_2d = np.outer(np.hamming(ty), np.hamming(tx))
        else:
            win_2d = np.ones((ty, tx))
            
        win_norm = np.sum(win_2d**2) * (dx * dy)  # Physical scale normalization

        # 5. Tile iteration and 2D FFT averaging
        psd_accumulator = np.zeros((ty, tx), dtype=np.float64)
        tile_count = 0

        for y in range(0, ny - ty + 1, step_y):
            for x in range(0, nx - tx + 1, step_x):
                tile = arr[y:y+ty, x:x+tx]
                
                # Remove DC offset per tile
                tile_detrend = tile - np.mean(tile)
                
                # Apply 2D spatial window
                tile_windowed = tile_detrend * win_2d
                
                # 2D FFT & Power calculation
                fft2_tile = np.fft.fft2(tile_windowed)
                power_tile = (np.abs(fft2_tile)**2) / win_norm
                
                psd_accumulator += power_tile
                tile_count += 1

        if tile_count == 0:
            raise ValueError(f"Tile size {tile_size} is larger than input array shape ({ny}, {nx})!")

        # 6. Average across tiles and shift zero-frequency (DC) component to center
        psd_avg = psd_accumulator / tile_count
        psd_shifted = np.fft.fftshift(psd_avg)

        # Convert to dB scale if requested
        if return_db:
            psd_output = 10 * np.log10(psd_shifted + 1e-12)
            var_name = 'welch_psd_db'
            unit_label = 'dB'
        else:
            psd_output = psd_shifted
            var_name = 'welch_psd'
            unit_label = f"({da.attrs.get('units', 'intensity')})^2 m^2"

        # 7. Calculate centered frequency axes (cycles per spatial unit)
        freq_y = np.fft.fftshift(np.fft.fftfreq(ty, d=dy))
        freq_x = np.fft.fftshift(np.fft.fftfreq(tx, d=dx))

        # 8. Build xarray.DataArray
        psd_da = xr.DataArray(
            psd_output,
            coords={'freq_y': freq_y, 'freq_x': freq_x},
            dims=['freq_y', 'freq_x'],
            name=var_name
        )
        
        # Preserve metadata attributes
        psd_da.attrs['units'] = unit_label
        psd_da.freq_y.attrs['units'] = '1/m'
        psd_da.freq_x.attrs['units'] = '1/m'
        psd_da.attrs['tile_size'] = f"{ty}x{tx}"
        psd_da.attrs['overlap'] = f"{int(overlap*100)}%"
        psd_da.attrs['num_tiles_averaged'] = tile_count

        return psd_da


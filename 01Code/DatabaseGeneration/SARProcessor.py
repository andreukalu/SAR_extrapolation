# IMPORTS
import netCDF4 as nc
import os
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter
import glob
import pickle
import numpy as np

# SAR file processor class
"""
    This class extracts useful information from pre-processed L-1 SLC Sentinel-1 AB data.
    
    All SAR files to be processed by this class should be pre-processed with SNAP software according to graph_merge_AS_remote.xml
    They should be in netcdf format.

    It can take as input a single SAR file or a folder where all files are stored.

    The available computations are:
    - Selection of tile of interest within the SAR image
        - Description: Select a tile of interest as set by lat, lon coordinates and width and height.
        - Function: obtain_target_tile
    - Computation of the 2D power spectral density
        - Description: Compute the 2D PSD of the selected tile of interest. It requires the tile selection first.
        - Function : compute_fft_2D, and compute_welch_2D
    - Removal of solid objects
        - Description: Application of a false alarm filter in order to remove targets such as ships or wind turbines from the SAR image.
        - Function: filter_objects
    - Intersection with FINO1 data
        - Description: Find the FINO1 data record corresponding to the SAR image measurement time
        - Function: get_closest_measurement
"""
class SARProcessor:

    def __init__(self,sar_src_path,sar_dst_path='',sar_file_name='',lat=0,lon=0,width=3500,height=3500):
        """
        Class constructor.

        Params:
        sar_src_path : String
            Path to the folder where all pre-processed SLC products are stored
        sar_dst_path : String (optional)
            Path to the folder where all products processed by this class will be stored if saved by write_pickle function.
        sar_file_name : String (optional)
            Path to a file to be processed if only a single one is to be analyzed.
        lat : float
            Target latitude (deg). All computations will be centered around this latitude.
        lon : float
            Target longitude (deg). All computations will be centered around this longitude.
        width : int
            Target tile width in pixels
        height : int
            Target tile height in pixels
        """

        # Save paths as attributes
        self.src_path = sar_src_path
        self.sar_dst_path = sar_dst_path
        self.file_name = sar_file_name
        self.file_path = os.path.join(sar_src_path,sar_file_name)

        # Create SAR dst directory
        dirname, fname = os.path.split(self.sar_dst_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # Add target coordinates
        self.lat = lat
        self.lon = lon

        # Add the width and height of the analysis window
        self.width = width
        self.height = height

    ############## METHODS ############
    def process_sar_files(self):
        """
            Process the SAR .nc files within src_path. Tiles at the coordinates of interest with dimensions
            width and height are cutted out from the complete image, thus reducing the dataset weight.
            
            IMPORTANT! this function requires that fino_path is defined and pointing to the FINO1 dataframe
        """

        # Get all the available files in src_path
        files = glob.glob(os.path.join(self.src_path,'*.nc'))

        # Process each .nc SAR measurement file
        for file in files:
            print(f'Processing file {file}')
            try:
                # Read the .nc file
                self.read_file(file)

                # Cut the target tile to be processed
                self.obtain_target_tile()

                # Filter out objects in the tile such as ships or wind turbines
                self.filter_objects(num_guard=20, num_ref=20, pfa=1e-3)

                # Compute the 2D PSDs of the tile using the Welch method and periodogram method
                self.compute_welch_2D(tile_size=(128, 128), overlap=0.5, window='hamming', return_db=True)
                self.compute_fft_2D()

                # Delete the dataset containing the whole SAR image and only retain the cutted tile
                del self.ds

                # Save the processed tile
                self.write_product(os.path.basename(file).split('.')[0])
            except:
                print(f'Couldnt process file {file}')

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

    def write_product(self,filename):
        """
            Function to save processed tiles as pickles
        
        Params:
        filename : String
            File name of the pickle to be saved.
        """

        # Create the pickle path
        path = os.path.join(self.sar_dst_path,filename+'.nc')

        # Add psd to the tile
        self.tile['psd'] = self.psd

        # Save the tile pickle
        self.tile.to_netcdf(path)
             
    def read_file(self, file_path=''):
        """
            Function to read a SAR file as an xarray dataset.
        
        Params:
        file_path : String
            path of the .nc SAR file to be read
        Returns:
        self.ds : xr.Dataset
            Dataset containing all the information in the SAR .nc file
        """

        # Check if the input file_path is to be used, or the on in the class attributes
        if file_path == '':
            self.ds = xr.open_dataset(self.file_path)
        else:
            self.ds = xr.open_dataset(os.path.join(self.src_path,file_path))

        # Add icident angle and platform heading for easy access
        self.ds['incident_angle'] = self.ds['incident_angle']
        self.ds['platform_heading'] = [
            val for key, val in self.ds.metadata.attrs.items() if key.endswith("platformHeading")
        ]
        self.ds['platform_heading'] = self.ds['platform_heading'][0]
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
        """
        Function to obtain the minimum distance of an input dataset to the target latitude and longitude

        Params:
        ds : xr.Dataset
            The dataframe containing the SAR-measured data to carry out the minimum distance search
        target_lan : float
            Target latitude (deg)
        target_lon : float
            Target longitude (deg)

        Returns:
        x_idx : int
            Index in the x dimension of the ds corresponding to the minimum distance to target coordinates
        y_idx : int
            Index in the y dimension of the ds corresponding to the minimum distance to target coordinates
        """
        # Calculate squared distance to target for every grid point
        distance = self.haversine_distance(ds.lat, ds.lon, target_lat, target_lon)

        # Find the 2D indices of the minimum distance
        min_dist_idx = distance.argmin(dim=['y', 'x'])

        y_idx = int(min_dist_idx['y'])
        x_idx = int(min_dist_idx['x'])

        return x_idx, y_idx

    def find_coordinates_index(self):
        """
            Obtain the indexes x and y of the SAR-measured image closest to self.lat and self.lon.
            Due to the large size of the images, a recursive search is carried out, decimating the image at each step
            for faster convergence.        
        """
        # Get the image sizes
        x_max_len = self.ds.sizes['x']
        y_max_len = self.ds.sizes['y']

        # Coarse Search (decimate by a factor 1000)
        skip_1 = 1000
        ds_step1 = self.ds.isel(x=slice(0, None, skip_1), y=slice(0, None, skip_1))
        x_idx, y_idx = self.find_minimum_distance(ds_step1, self.lat, self.lon)
        
        # Scale back to full dataset coordinates
        x_abs = x_idx * skip_1
        y_abs = y_idx * skip_1

        # Define bounded box around the coarse estimate
        x_min = max(0, x_abs - skip_1)
        x_max = min(x_max_len, x_abs + skip_1 + 1)
        y_min = max(0, y_abs - skip_1)
        y_max = min(y_max_len, y_abs + skip_1 + 1)

        # Medium Search (decimate by a factor 100)
        skip_2 = 100
        ds_step2 = self.ds.isel(x=slice(x_min, x_max, skip_2), y=slice(y_min, y_max, skip_2))
        x_idx, y_idx = self.find_minimum_distance(ds_step2, self.lat, self.lon)
        
        # Map back to full dataset coordinates
        x_abs = x_min + (x_idx * skip_2)
        y_abs = y_min + (y_idx * skip_2)

        # Define fine bounded box
        x_min = max(0, x_abs - skip_2)
        x_max = min(x_max_len, x_abs + skip_2 + 1)
        y_min = max(0, y_abs - skip_2)
        y_max = min(y_max_len, y_abs + skip_2 + 1)

        # Fine Search (full resolution)
        ds_reduced = self.ds.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
        x_idx, y_idx = self.find_minimum_distance(ds_reduced, self.lat, self.lon)
        
        # Final global indices
        final_x = x_min + x_idx
        final_y = y_min + y_idx

        self.x_coordinate = final_x
        self.y_coordinate = final_y

    def obtain_target_tile(self):
        """
            Function to retrieve a tile of SAR-measured values at the target coordinates of the specified width and height at the class constructor
        
        Return:
        self.tile : xr.Dataset
            Dataset containing the target tile of the image.
        """
        self.find_coordinates_index()

        self.tile = self.ds.sel(x=slice(self.x_coordinate-int(self.width/2),self.x_coordinate+int(self.width/2)),y=slice(self.y_coordinate-int(self.height/2),self.y_coordinate+int(self.height/2)))

        return self.tile

    def cfar_mask_2d_numpy(self, arr, num_guard=3, num_ref=6, pfa=1e-4):
        """
        NumPy 2D CA-CFAR implementation operating on spatial arrays for target detection.

        Params:
        arr : xr.Array
            Input SAR image array
        num_guard : integer
            guard pixels for the filter
        num_ref : integer
            reference pixels for the filter
        pfa : float
            Probability of false alarm for the filter.

        Return:
        arr_clean : xr.Array
            Array after removal of objects detected by the CA-CFAR filter
        target_mask : xr.Array
            Mask with the identified objects 
        """
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


    def filter_objects(self, var_name='Sigma0_VV', spatial_dims=None, 
                            num_guard=20, num_ref=20, pfa=1e-3):
        """
        Apply CFAR Target Masking directly to the previously cropped tile.
        Auto-detects spatial dimensions if spatial_dims is None.

        Params:
        var_name : String
            Name of the variable to be processed by the object detection filter removal.
        spatial_dims : Tuple of strings
            Name of the spatial dimensions of the image.
        num_guard : integer
            guard pixels for the filter
        num_ref : integer
            reference pixels for the filter
        pfa : float
            Probability of false alarm for the filter.    

        Return:
        clean_ds : xr.Dataset
            The dataset containing the tile with the objects removed.   
        """
        da = self.tile[var_name] if isinstance(self.tile, xr.Dataset) else self.tile

        # Auto-detect spatial dimensions in exact dataset order
        if spatial_dims is None:
            spatial_dims = da.dims[-2:]
        
        # Ensure spatial_dims match the exact order in the DataArray
        spatial_dims = tuple(d for d in da.dims if d in spatial_dims)

        def _ufunc_wrapper(arr):
            cleaned, mask = self.cfar_mask_2d_numpy(
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

        if isinstance(self.tile, xr.Dataset):
            ds_out = self.tile.copy()
            ds_out[da_clean.name] = da_clean
            ds_out[da_mask.name] = da_mask
            self.tile = ds_out
            return ds_out
        else:
            return xr.merge([da_clean, da_mask])

    def compute_fft_2D(self, var_name='Sigma0_VV_no_targets', spatial_dims=('y', 'x'), filter=True, normalize=True):
            """
            Computes 2D FFT on an xarray DataArray and returns a DataArray with frequency axes.
    
            Params:
            var_name : String
                Name of the variable to be processed by the object detection filter removal.
            spatial_dims : Tuple of strings
                Name of the spatial dimensions of the image.
            
            Returns:
            fft_da : xr.Array
                The computed PSD with the periodogram method
            """
            dim_y, dim_x = spatial_dims
            
            # Fill NaNs before FFT (FFTs cannot process NaNs)
            da_filled = self.tile[var_name].fillna(np.mean(self.tile[var_name]))
    
            # Subtract mean component
            da_demeaned = da_filled - np.mean(da_filled)

            # Get the signal energy
            total_energy_spatial = np.sum(da_demeaned.values**2)

            # Calculate sampling intervals (dx, dy) in physical units
            dy = np.abs(np.diff(self.tile[dim_y].values)[0])
            dx = np.abs(np.diff(self.tile[dim_x].values)[0])
    
            # Assumed values
            dy = self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing']
            dx = self.tile.metadata.attrs['Abstracted_Metadata:range_spacing']
    
            if filter == True:
                # 3. Create and apply a 2D Hanning window to suppress boundary artifacts
                Ny, Nx = da_demeaned.shape
                win_y = np.hanning(Ny)
                win_x = np.hanning(Nx)
                window_2d = np.outer(win_y, win_x)
                
                da_demeaned = da_demeaned * window_2d
    
            # 2D FFT computation
            fft_vals = np.fft.fftshift(np.fft.fft2(da_demeaned))
            psd_raw = np.abs(fft_vals)**2
    
            # Normalize PSD to the maximum value if desired
            if normalize == True:
                psd_max = np.max(psd_raw)
                if psd_max > 0:
                    psd_norm = psd_raw / psd_max
                else:
                    psd_norm = psd_raw
                    
                self.tile['fft_normalization_factor'] = psd_max
            else:
                psd_norm = psd_raw
    
            # Convert to dB (Normalized max will be 0 dB)
            power_db = 10 * np.log10(psd_norm + 1e-8)
            
            # Compute centered frequency coordinates (cycles per spatial unit)
            freq_y = np.fft.fftshift(np.fft.fftfreq(self.tile[dim_y].size, d=dy))
            freq_x = np.fft.fftshift(np.fft.fftfreq(self.tile[dim_x].size, d=dx))
            
            # Construct output DataArray
            fft_da = xr.DataArray(
                power_db,
                coords={'freq_y': freq_y, 'freq_x': freq_x},
                dims=['freq_y', 'freq_x'],
                name='power_spectrum_db'
            )
            fft_da.freq_y.attrs['units'] = '1/m'
            fft_da.freq_x.attrs['units'] = '1/m'
    
            self.psd = fft_da
            self.tile['spatial_energy'] = total_energy_spatial
            
            return fft_da
    

    def compute_welch_2D(self, var_name='Sigma0_VV_no_targets', spatial_dims=('y', 'x'), tile_size=(256, 256), overlap=0.5, window='hanning', return_db=True):
        """
        Computes 2D Welch Power Spectral Density (PSD) on an xarray DataArray 
        and returns a DataArray with centered spatial frequency axes (cycles per unit) using the Welch method.

        Params:
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
        xr.DataArray
            2D Welch PSD with centered frequency coordinates (freq_y, freq_x).
        """
        dim_y, dim_x = spatial_dims
        
        # 1. Fill NaNs with mean to prevent FFT corruption
        da_filled = self.tile[var_name].fillna(self.tile[var_name].mean().item()) - np.mean(self.tile[var_name].fillna(self.tile[var_name].mean().item()))
        arr = da_filled.values.astype(np.float64)
        
        ny, nx = arr.shape
        ty, tx = tile_size

        # 2. Extract spatial resolution (dx, dy) from coordinate vectors
        dy = np.abs(np.diff(self.tile[dim_y].values)[0])
        dx = np.abs(np.diff(self.tile[dim_x].values)[0])
        
        # Assumed values
        dy = self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing']
        dx = self.tile.metadata.attrs['Abstracted_Metadata:range_spacing']

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
                tile_detrend = tile #- np.mean(tile)
                
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
            unit_label = f"({self.tile_filtered.attrs.get('units', 'intensity')})^2 m^2"

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

        self.psd = psd_da
        
        return psd_da

    def get_closest_measurement(self, datetime_col='TIME'):
        """
        Finds the row in FINO1 dataframe self.df where datetime_col is closest to target_time_str.
        Requires that the FINO1 dataframe is read with self.read_fino_file()

        Params:
        datetime_col : String
            Name of the datetime column in FINO1 dataset
        
        Returns:
        self.tile : xr.Dataset
            The cropped SAR tile with the added FINO1 information corresponding to the SAR measurement time.
        """
        target_time_str = self.tile.start_date

        # Parse target string into a pandas Timestamp
        target_dt = pd.to_datetime(target_time_str)
        
        # Ensure the dataframe column is datetime type
        dt_series = pd.to_datetime(self.df[datetime_col])
        
        # Find index of minimum absolute time difference
        closest_idx = (dt_series - target_dt).abs().idxmin()
        
        # Return the row as a DataFrame (or Series via df.loc[closest_idx])
        row = self.df.loc[[closest_idx]]
    
        # Convert the single row into a Series
        row_series = row.iloc[0]
    
        # Add each item from the row into your Dataset
        for col_name, value in row_series.items():
            # If the value is a pandas Timestamp, convert to string or numpy datetime
            if isinstance(value, pd.Timestamp):
                value = str(value)
                
            self.tile[col_name] = value
    
        return self.tile

    ########### TODO: functions to implement ####################

    # def retrieve_stability_from_psd(self):

    # def wind_retrieval(self):

    ########### HELP: functions that can be used ################
    # def calc_sigma0_cmod5_n(self, v, phi, theta):
    #     """Calculates CMOD5n normalized radar backscatter (linear).

    #     Parameters:
    #         v: Wind speed [m/s] (>= 0)
    #         phi: Relative wind direction [deg] (angle between azimuth and wind
    #         direction)
    #         theta: Incidence angle [deg]

    #     Returns:
    #         CMOD5_N: Normalized backscatter sigma0 (linear scale)
    #     """

    #     # 1-based indexing added to match Fortran C(1) .. C(28)
    #     C = np.array(
    #         [
    #             0.0,  # Index 0 unused to maintain 1-based Fortran indexing
    #             -0.6878,
    #             -0.7957,
    #             0.3380,
    #             -0.1728,
    #             0.0000,
    #             0.0040,
    #             0.1103,
    #             0.0159,
    #             6.7329,
    #             2.7713,
    #             -2.2885,
    #             0.4971,
    #             -0.7250,
    #             0.0450,
    #             0.0066,
    #             0.3222,
    #             0.0120,
    #             22.7000,
    #             2.0813,
    #             3.0000,
    #             8.3659,
    #             -3.3428,
    #             1.3236,
    #             6.2437,
    #             2.3893,
    #             0.3249,
    #             4.1590,
    #             1.6930,
    #         ]
    #     )

    #     DTOR = 57.29577951
    #     THETM = 40.0
    #     THETHR = 25.0
    #     ZPOW = 1.6

    #     Y0 = C[19]
    #     PN = C[20]
    #     A = C[19] - (C[19] - 1.0) / C[20]
    #     B = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

    #     # Angles
    #     FI = np.radians(phi)  # equivalent to phi / DTOR
    #     CSFI = np.cos(FI)
    #     CS2FI = 2.0 * CSFI * CSFI - 1.0

    #     X = (theta - THETM) / THETHR
    #     XX = X * X

    #     # B0: Function of wind speed and incidence angle
    #     A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
    #     A1 = C[5] + C[6] * X
    #     A2 = C[7] + C[8] * X

    #     GAM = C[9] + C[10] * X + C[11] * XX
    #     S0 = C[12] + C[13] * X

    #     S = A2 * v
    #     A3 = 1.0 / (1.0 + np.exp(-np.maximum(S, S0)))

    #     # Piecewise condition for S < S0
    #     if np.ndim(S) > 0:
    #         mask = S < S0
    #         A3[mask] = A3[mask] * (S[mask] / S0[mask]) ** (
    #             S0[mask] * (1.0 - A3[mask])
    #         )
    #     else:
    #         if S < S0:
    #             A3 = A3 * (S / S0) ** (S0 * (1.0 - A3))

    #     B0 = (A3**GAM) * (10.0 ** (A0 + A1 * v))

    #     # B1: Function of wind speed and incidence angle
    #     B1 = C[15] * v * (0.5 + X - np.tanh(4.0 * (X + C[16] + C[17] * v)))
    #     B1 = C[14] * (1.0 + X) - B1
    #     B1 = B1 / (np.exp(0.34 * (v - C[18])) + 1.0)

    #     # B2: Function of wind speed and incidence angle
    #     V0 = C[21] + C[22] * X + C[23] * XX
    #     D1 = C[24] + C[25] * X + C[26] * XX
    #     D2 = C[27] + C[28] * X

    #     V2 = (v / V0) + 1.0

    #     # Piecewise condition for V2 < Y0
    #     if np.ndim(V2) > 0:
    #         mask2 = V2 < Y0
    #         V2[mask2] = A + B * (V2[mask2] - 1.0) ** PN
    #     else:
    #         if V2 < Y0:
    #             V2 = A + B * (V2 - 1.0) ** PN

    #     B2 = (-D1 + D2 * V2) * np.exp(-V2)

    #     # CMOD5_N: Combine the three Fourier terms
    #     CMOD5_N = B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW

    #     return CMOD5_N

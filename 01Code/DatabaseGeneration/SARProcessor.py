# IMPORTS
import netCDF4 as nc
import os
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter
from scipy.signal import detrend
from scipy.optimize import least_squares
import scipy
import traceback
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

    def __init__(self,sar_src_path,sar_dst_path='',sar_file_name='',sar_statistics_path='',lat=0,lon=0,width=3500,height=3500):
        """
        Class constructor.

        Params:
        sar_src_path : String
            Path to the folder where all pre-processed SLC products are stored
        sar_dst_path : String (optional)
            Path to the folder where all products processed by this class will be stored if saved by write_pickle function.
        sar_file_name : String (optional)
            Path to a file to be processed if only a single one is to be analyzed.
        sar_statistics_path : String (optional)
            Path to the folder where all statistics processed by this class will be stored if saved by write_pickle function.
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
        self.sar_statistics_path = sar_statistics_path

        # Create SAR dst directory if passed as input
        if self.sar_dst_path != '':
            dirname, fname = os.path.split(self.sar_dst_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

        # Create SAR statistics directory if passed as input
        if self.sar_statistics_path != '':
            dirname, fname = os.path.split(self.sar_statistics_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

        # Add target coordinates
        self.lat = lat
        self.lon = lon

        # Add the width and height of the analysis window
        self.width = width
        self.height = height

    ############## METHODS ############
    def process_sar_files(self, psd = True, statistics = True, autocorr = False, wavelet = False):
        """
            Process the SAR .nc files within src_path. Tiles at the coordinates of interest with dimensions
            width and height are cutted out from the complete image, thus reducing the dataset weight.
            
            IMPORTANT! this function requires that fino_path is defined and pointing to the FINO1 dataframe

            Params:
            psd : Bool (optional)
                Flag for PSD computation through Welch method
            statistics : Bool (optional)
                Flag for PSD statistics computation
            autocorr : Bool (optional)
                Flag for autocorrelation computation
            wavelet : Bool (optional)
                Flat for wavelet transform computation
        """
        print('Processing all SAR files')
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

                if psd: # Compute the 2D PSDs of the tile using the Welch method
                    self.compute_welch_2D(tile_size=(1024, 1024), overlap=0.4, window='hamming', return_db=True)

                if statistics:
                    self.compute_PSD_statistics(f_bins=np.arange(1e-4, 5e-3, 1e-4),theta_bins=np.deg2rad(np.arange(0, 181, 15)))

                if wavelet: # Compute the Wavelet transform in 2 dimensions
                    self.compute_wavelet_2D()

                if autocorr: # Compute the autocorrelation of the tile 
                    self.compute_autocorr_2D()

                # Delete the dataset containing the whole SAR image and only retain the cutted tile
                del self.ds
                
                # Save the processed tile
                self.write_product(os.path.basename(file).split('.')[0], psd=psd, autocorr=autocorr, wavelet=wavelet)

            except Exception as e:
                print(f'Could not process file {file}')
                print(f'Error: {type(e).__name__}: {e}')
                traceback.print_exc()
                continue

    def process_single_sar_file(self,filename,psd=True,statistics=True,autocorr=False,wavelet=False):
            """
                Process a single SAR .nc file within src_path. Tiles at the coordinates of interest with dimensions
                width and height are cutted out from the complete image, thus reducing the dataset weight.
                
                IMPORTANT! this function requires that fino_path is defined and pointing to the FINO1 dataframe

                Params:
                filename : String
                    Filename of the file SAR file to be processed
                psd : Bool (optional)
                    Flag for PSD computation through Welch method
                autocorr : Bool (optional)
                    Flag for autocorrelation computation
                wavelet : Bool (optional)
                    Flat for wavelet transform computation
            """
    
            # Get all the available files in src_path
            file = os.path.join(self.src_path,filename)
    
            # Process each .nc SAR measurement file
            print(f'Processing file {file}')
            try:
                # Read the .nc file
                self.read_file(file)

                # Cut the target tile to be processed
                self.obtain_target_tile()

                # Filter out objects in the tile such as ships or wind turbines
                self.filter_objects(num_guard=20, num_ref=20, pfa=1e-3)

                if psd: # Compute the 2D PSD of the tile using the Welch method
                    self.compute_welch_2D(tile_size=(1024, 1024), overlap=0.4, window='hamming', return_db=True)

                if statistics: # Compute the PSD statistics
                    
                    if psd == False: # If PSD is not to be computed, try to obtain the already processed PSD
                        self.read_single_sar_postprocessed_file(os.path.basename(file).split('.')[0] + '.nc')
                    self.compute_PSD_statistics(f_bins=np.arange(1e-4, 5e-3, 1e-4),theta_bins=np.deg2rad(np.arange(0, 181, 15)))

                if wavelet: # Compute the wavelet transform of the tile
                    self.compute_wavelet_2D()

                if autocorr: # Compute the autocorrelation of the tile
                    self.compute_autocorr_2D()

                # Delete the dataset containing the whole SAR image and only retain the cutted tile
                del self.ds

                # Save the processed tile
                self.write_product(os.path.basename(file).split('.')[0], psd=psd, statistics=statistics,autocorr=autocorr, wavelet=wavelet)
            
            except Exception as e:
                print(f'Could not process file {file}')
                print(f'Error: {type(e).__name__}: {e}')
                traceback.print_exc()

    def read_single_sar_postprocessed_file(self,filename):
        """
        Function to load the tile of an already processed SAR file.

        Params:
            filename : String
                SAR product filename
        """

        # Get the SAR postprocessed files path
        path = os.path.join(self.sar_dst_path,filename)

        # Load the product
        tile = xr.open_dataset(path)
        self.tile = tile
        self.psd = self.tile['psd']
        self.psd_max = self.tile['psd_max']
        self.power = self.tile['power']
        self.phase_coherence = self.tile['phase_coherence']

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

    def write_product(self,filename,psd=True,statistics=True,autocorr=False,wavelet=False):
        """
            Function to save processed tiles as pickles
        
        Params:
        filename : String
            File name of the pickle to be saved.
        psd : Bool (optional)
            Flag for the addition of PSD product information into the product written
        statistics : Bool (optional)
            Flag for the addition of PSD 2nd order statistics products into the product written
        autocorr : Bool (optional)
            Flag for the addition of autocorrelation product information into the product written
        wavelet : Bool (optional)
            Flag for the addition of the wavelet transform information into the product written
        """

        # Create the pickles paths
        # If the psd, autocorrelation or wavelet are to be added, create the sar_dst_path
        if psd == True or autocorr == True or wavelet == True:
            full_path = os.path.join(self.sar_dst_path,filename+'.nc')

        # If the 2nd order statistics are to be saved, create a separate pickle path for it
        if statistics == True:
            statistics_path = os.path.join(self.sar_statistics_path,filename+'_statistics.nc')

        # Obtain the SAR measured tile as the base dataset.
        self.stats = self.tile

        # Add psd to the tile
        if psd:
            self.tile['psd'] = self.psd
            self.tile['psd_max'] = self.psd_max
            self.tile['power'] = self.power
            self.tile['phase_coherence'] = self.phase_coherence

        if statistics:
            self.tile['phase_coherence'] = self.phase_coherence
            self.tile['max_component'] = (('freq', 'theta'), self.max_component)
            self.tile['min_component'] = (('freq', 'theta'), self.min_component)
            self.tile['mean_component'] = (('freq', 'theta'), self.mean_component)
            self.tile['max_component_coherence'] = (('freq', 'theta'), self.max_component_coherence)
            self.tile['anisotropy'] = self.anisotropy
            self.tile['phase_coherence_max'] = (('freq', 'theta'), self.phase_coherence_max)
            self.tile['max_phase_coherence'] = (('freq', 'theta'), self.max_phase_coherence)
            self.tile['min_phase_coherence'] = (('freq', 'theta'), self.min_phase_coherence)
            self.tile['mean_phase_coherence'] = (('freq', 'theta'), self.mean_phase_coherence)
            self.tile['max_significance'] = (('freq', 'theta'), self.max_significance)
            self.tile['mean_significance'] = (('freq', 'theta'), self.mean_significance)
            self.tile['min_significance'] = (('freq', 'theta'), self.min_significance)
            self.tile['psd_max'] = self.psd_max
            self.tile['A'] = self.A
            self.tile['an1'] = self.an1
            self.tile['f01'] = self.f01
            self.tile['f02'] = self.f02
            self.tile['detrended_psd'] = (('freq_x', 'freq_y'), self.detrended_psd)
            self.tile['model_psd'] = (('freq_x', 'freq_y'), self.model_psd)
            self.tile['fit_rmse'] = self.fit_rmse
            self.stats['f_bins'] = self.f_bins
            self.stats['theta_bins'] = self.theta_bins
            if getattr(self, "tile", None) is not None and "psd" in self.stats:
                self.stats = self.stats.drop_vars("psd")
            if getattr(self, "tile", None) is not None and "Sigma0_VH" in self.stats:
                self.stats = self.stats.drop_vars("Sigma0_VH")
            if getattr(self, "tile", None) is not None and "Sigma0_VV" in self.stats:
                self.stats = self.stats.drop_vars("Sigma0_VV")
            if getattr(self, "tile", None) is not None and "Sigma0_VV_no_targets" in self.stats:
                self.stats = self.stats.drop_vars("Sigma0_VV_no_targets")

        # Add autocorr to the tile
        if autocorr:
            self.tile['autocorr'] = self.autocorr

        # Add wavelet transform to the tile
        if wavelet:
            self.tile.update(self.stats)

        if psd == True or autocorr == True or wavelet == True:
            # Save the tile pickle
            self.tile.to_netcdf(full_path)

        if statistics == True:
            # Save the statistics pickle
            self.stats.to_netcdf(statistics_path)
             
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
    

    def compute_welch_2D(self, var_name='Sigma0_VV_no_targets', tile_size=(256, 256), overlap=0.5, window='hanning', return_db=True, normalize=True):
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
        
        #Fill NaNs with mean to prevent FFT corruption
        da_filled = self.tile[var_name].fillna(self.tile[var_name].mean().item()) - np.mean(self.tile[var_name].fillna(self.tile[var_name].mean().item()))
        arr = da_filled.values.astype(np.float64)

        # Get the tile dimensions
        ny, nx = arr.shape
        ty, tx = tile_size
        
        # Extract spatial resolution (dx, dy) from coordinate vectors
        dy = self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing']
        dx = self.tile.metadata.attrs['Abstracted_Metadata:range_spacing']

        # Calculate step sizes based on overlap percentage
        step_y = max(1, int(ty * (1.0 - overlap)))
        step_x = max(1, int(tx * (1.0 - overlap)))

        # Construct 2D tapering window and energy normalization factor
        if window == 'hanning':
            win_2d = np.outer(np.hanning(ty), np.hanning(tx))
        elif window == 'hamming':
            win_2d = np.outer(np.hamming(ty), np.hamming(tx))
        else:
            win_2d = np.ones((ty, tx))
            
        win_norm = np.sum(win_2d**2) * (dx * dy)  # Physical scale normalization

        # Tile iteration and 2D FFT averaging
        psd_accumulator = np.zeros((ty, tx), dtype=np.float64)

        # Preallocate phase array
        phi2_accumulator = np.zeros((ty, tx), dtype=np.complex128)

        # Start with tile 0
        tile_count = 0

        # Iterate over all the tiles
        for y in range(0, ny - ty + 1, step_y):
            for x in range(0, nx - tx + 1, step_x):

                # Select the tile to be analyzed
                tile = arr[y:y+ty, x:x+tx]
                
                # Remove DC offset per tile through a least squares fitting of a plane and its subtraction
                yy, xx = np.mgrid[0:ty, 0:tx]
                A = np.column_stack([np.ones(ty*tx), yy.ravel(), xx.ravel()])
                coeffs, *_ = np.linalg.lstsq(A, tile.ravel(), rcond=None)
                plane = coeffs[0] + coeffs[1]*yy + coeffs[2]*xx

                tile_detrend = tile - plane
                # Apply 2D spatial window
                tile_windowed = tile_detrend * win_2d
                
                # 2D FFT, Power spectral density, and phase coherence calculation
                fft2_tile = np.fft.fft2(tile_windowed)
                # Get the tiles centers
                yc = y + ty / 2
                xc = x + tx / 2
                # Get the phase of the tile PSD and subtract the phase ramp
                phi2_tile = np.angle(fft2_tile) + 2*np.pi*(np.fft.fftfreq(ty)[:, None] * yc + np.fft.fftfreq(tx)[None, :] * xc)
                # Obtain the power of the tile
                power_tile = (np.abs(fft2_tile)**2) / win_norm

                # Accumulate the power and the phase
                psd_accumulator += power_tile
                phi2_accumulator += np.exp(1j * phi2_tile)

                # Update the tile
                tile_count += 1

        if tile_count == 0:
            raise ValueError(f"Tile size {tile_size} is larger than input array shape ({ny}, {nx})!")

        # Average across tiles and shift zero-frequency (DC) component to center
        psd_avg = psd_accumulator / tile_count
        phase_coherence = np.abs(phi2_accumulator / tile_count)
        psd_shifted = np.fft.fftshift(psd_avg)
        phase_coherence_shifted = np.fft.fftshift(phase_coherence)

        if normalize: # If normalize flag True, normalize the PSD and save the max
            self.psd_max = np.max(psd_shifted)
            self.power = np.sum(psd_shifted)
            psd_shifted = psd_shifted/np.max(psd_shifted)
        else: # If normalize flag False, set the psd_max output to nan
            self.psd_max = np.nan
            self.power = np.sum(psd_shifted)

        # Convert to dB scale if requested
        if return_db:
            psd_output = 10 * np.log10(psd_shifted + 1e-12)
            var_name = 'welch_psd_db'
            unit_label = 'dB'
        else:
            psd_output = psd_shifted
            var_name = 'welch_psd'
            unit_label = f"({self.tile.attrs.get('units', 'intensity')})^2 m^2"

        # Calculate centered frequency axes (cycles per spatial unit)
        freq_y = np.fft.fftshift(np.fft.fftfreq(ty, d=dy))
        freq_x = np.fft.fftshift(np.fft.fftfreq(tx, d=dx))

        # Build xarray.DataArray for the PSD
        psd_da = xr.DataArray(
            psd_output,
            coords={'freq_y': freq_y, 'freq_x': freq_x},
            dims=['freq_y', 'freq_x'],
            name=var_name
        )

        # Build xarray.DataArray for the phase coherence
        phase_coherence_da = xr.DataArray(
            phase_coherence_shifted,
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

        # Set the processed information as class attributes
        self.phase_coherence = phase_coherence_da
        self.psd = psd_da
        
        return psd_da

    def compute_PSD_statistics(self, f_bins=np.arange(1e-4, 5e-3, 1e-4),theta_bins=np.deg2rad(np.arange(0, 181, 15))):
        
        # Compute the statistics of the PSD
        max_component, min_component, mean_component, max_component_coherence = self.compute_max_psd_component_f(f_bins,theta_bins)
        anisotropy = self.compute_anisotropy_f(f_bins)
        phase_coherence_max, max_phase_coherence, min_phase_coherence, mean_phase_coherence = self.compute_phase_coherence_f(f_bins,theta_bins)
        max_significance, mean_significance, min_significance = self.compute_component_significance(f_bins,theta_bins)
        A, an1, f01, f02, detrended_psd, model_psd, fit_rmse = self.fit_psd_trend()
        
        # Set the processed information as class attributes
        self.max_component = max_component
        self.min_component = min_component
        self.mean_component = mean_component
        self.max_component_coherence = max_component_coherence
        self.anisotropy = anisotropy
        self.phase_coherence_max = phase_coherence_max
        self.max_phase_coherence = max_phase_coherence
        self.min_phase_coherence = min_phase_coherence
        self.mean_phase_coherence = mean_phase_coherence
        self.max_significance = max_significance
        self.mean_significance = mean_significance
        self.min_significance = min_significance
        self.A = A
        self.an1 = an1
        self.f01 = f01
        self.f02 = f02
        self.detrended_psd = detrended_psd
        self.model_psd = model_psd
        self.fit_rmse = fit_rmse
        self.f_bins = f_bins
        self.theta_bins = theta_bins

    def fit_psd_trend(self, f_lim = 1e-2, in_db=True):
        """
        Function to fit a model to the 2D PSD. A Bode-type function is fitted with a pole and a zero.
        The function is of the shape H(f) = A * ((f ** 2 + f01 ** 2)/(f ** 2 + f02 ** 2)) ** an1
        In log space, A is the noise level, f01 and f02 are the frequencies of the poles and zeros, 
        and an1 is the slope

        Params:
            f_lim : float
                Frequency limit of the fitting
            in_db : Bool
                Flag for indicating if the input PSD is in dBs 
        """ 

        # Get the frequency values and generate a 2D meshgrid
        fx = self.psd['freq_x'].values
        fy = self.psd['freq_y'].values
        FX, FY = np.meshgrid(fx, fy, indexing='xy')

        # Compute the radial frequency as the quadratic sum of f_x and f_y
        F = np.hypot(FX, FY)

        # Get the psd and phase coherence values from the class attributes
        psd = self.psd.values
        coh = self.phase_coherence.values

        # Mask out the PSD, F and phase coherence values at frequencies <= f_lim
        mask = F <= f_lim
        psd_fit = psd[mask]
        F_fit = F[mask]

        # convert to linear power once; fit in log space
        if in_db:
            psd_lin_fit = 10.0 ** (psd_fit / 10.0)
            psd_lin_all = 10.0 ** (psd / 10.0)
        else:
            psd_lin_fit = psd_fit
            psd_lin_all = psd

        good = (F_fit > 0) & np.isfinite(psd_lin_fit) & (psd_lin_fit > 0)
        f = F_fit[good].ravel()
        y = 10*np.log10(psd_lin_fit[good].ravel())     # log10 of measured power

        # Define the model function in dBs
        def residual(p):
            A, an1, f01, f02 = p
            model = 10*(np.log10(A) + an1*np.log10(f**2+f01**2) + (-an1)*np.log10(f**2+f02**2))
            return model - y

        # initial guess for fitting params
        f01 = 1e-5
        f02 = 1e-3
        an1_0 = -2
        A0 = np.min(10**(y/10.0))
        p0 = [A0, an1_0, f01, f02]

        # Fitting bounds definition
        bounds = ([0.0, -10.0, 0, 0],
                    [ 50.0, 0.0, 1e-2, 1e-2])

        # Fitting of the model to the PSD through least squares
        result = least_squares(
            residual, p0,
            bounds=bounds,
        )

        # Save the fitting RMSE
        fit_rmse = np.sqrt(np.mean(residual(result.x)**2))

        # Save the fitting result variables
        A, an1, f01, f02 = result.x

        # Obtain the modelled psd and detrended psd
        with np.errstate(divide='ignore', invalid='ignore'):
            model_psd = np.where(F > 0, A*(F**2+f01**2)**an1 * (F**2+f02**2)**(-an1), np.nan)
            detrended_psd = np.where(F > 0, psd_lin_all / model_psd, np.nan)

        if in_db:
            with np.errstate(divide='ignore', invalid='ignore'):
                detrended_psd = 10.0 * np.log10(detrended_psd)
                model_psd = 10.0 * np.log10(model_psd)

        return A, an1, f01, f02, detrended_psd, model_psd, fit_rmse
    
    def compute_max_psd_component_f(self, f_bins=np.arange(1e-4, 5e-3, 1e-4),theta_bins=np.deg2rad(np.arange(0, 181, 15))):
            """
            Obtains binned PSD statistics in bins of frequency and polar angle theta.
    
            Params:
            f_bins : np.Array
                Frequency bins
            theta_bins : np.Array
                Theta bins
            in_db : Bool
                Flag indicating if the input is in dBs
    
            Returns:
            max_component : float
                Maximum PSD component for each bin
            min_component : float
                Minimum PSD component for each bin
            mean_component : float
                Mean PSD component for each bin
            max_component_coherence : float
                PSD component with the max coherence
            """

            # Get fx and fy and compute the 2D meshgrid
            fx = self.psd['freq_x'].values
            fy = self.psd['freq_y'].values
            FX, FY = np.meshgrid(fx, fy, indexing='xy')

            # Compute the radial frequency and the polar angle
            F = np.sqrt(FX**2 + FY**2)
            THETA = np.mod(np.arctan2(FY, FX), np.pi)   # symmetry: [0, pi)

            # Get the PSD and phase coherence values
            psd_values = self.psd.values
            coh_values = self.phase_coherence.values

            # Get the number of bins
            n_f = len(f_bins) - 1
            n_t = len(theta_bins) - 1

            # Preallocate the output variables
            max_component           = np.full((n_f, n_t), np.nan)
            min_component           = np.full((n_f, n_t), np.nan)
            mean_component          = np.full((n_f, n_t), np.nan)
            max_component_coherence = np.full((n_f, n_t), np.nan)

            # For each frequency and theta bin
            for i, (f_min, f_max) in enumerate(zip(f_bins[:-1], f_bins[1:])):
                for j, (t_min, t_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):

                    # Get the bin mask
                    mask = (F >= f_min) & (F < f_max) & (THETA >= t_min) & (THETA < t_max)
                    if not np.any(mask):
                        continue

                    # Mask the PSD and phase coherence bins
                    psd_bin = psd_values[mask]
                    coh_bin = coh_values[mask]

                    # Compute the output statistics
                    max_component[i, j]  = np.max(psd_bin)                          # Bin max PSD
                    min_component[i, j]  = np.min(psd_bin)                          # Bin min PSD
                    mean_component[i, j] = 10*np.log10(np.mean(10**(psd_bin/10)))   # Bin mean PSD
                    max_component_coherence[i, j] = psd_bin[np.argmax(coh_bin)]     # Bin max phase coherence PSD value

            return max_component, min_component, mean_component, max_component_coherence

    def compute_component_significance(self, f_bins=np.arange(1e-4, 5e-3, 1e-4),theta_bins=np.deg2rad(np.arange(0, 181, 15)), in_db=True):
        """
        Computes the component significance in bins of f_bins and theta_bins.
        Component significance is defined as the product of PSD and phase coherence

        Params:
        f_bins : np.Array
            Frequency bins
        theta_bins : np.Array
            Theta bins
        in_db : Bool
            Flag indicating if the input is in dBs

        Returns:
        max_significance : float
            Max significance per analyzed bin
        mean_significance : float
            Mean significance per analyzed bin
        min_significanceropy : float
            Min significance per analyzed bin
        """

        # Get the fx, fy values and generate a meshgrid
        fx = self.psd['freq_x'].values
        fy = self.psd['freq_y'].values
        FX, FY = np.meshgrid(fx, fy, indexing='xy')

        # Get the radial frequency as the quadratic sum of fx and fy
        F = np.sqrt(FX**2 + FY**2)
        # Get the polar angle from the arctan of fx and fy
        THETA = np.mod(np.arctan2(FY, FX), np.pi)   # symmetry: [0, pi)

        # Convert the PSD values to linear and get the phase coherence
        if in_db:
            psd_values = 10**(self.psd.values/10)
        coh_values = self.phase_coherence.values

        # Compute the significance as the product of the PSD and phase coherence
        significance = 10*np.log10(psd_values*coh_values)

        # Get the number of f and theta bins
        n_f = len(f_bins) - 1
        n_t = len(theta_bins) - 1

        # Preallocate the output arrays
        max_significance    = np.full((n_f, n_t), np.nan)
        mean_significance   = np.full((n_f, n_t), np.nan)
        min_significance    = np.full((n_f, n_t), np.nan)

        # For each frequency and theta bins
        for i, (f_min, f_max) in enumerate(zip(f_bins[:-1], f_bins[1:])):
            for j, (t_min, t_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):

                # Get the mask of the frequency and theta bin
                mask = (F >= f_min) & (F < f_max) & (THETA >= t_min) & (THETA < t_max)
                if not np.any(mask):
                    continue

                # Mask the selected bin
                significance_bin = significance[mask]

                # Comput the output significance statistics
                max_significance[i, j]  = np.nanmax(significance_bin)
                mean_significance[i, j] = np.nanmean(significance_bin)
                min_significance[i, j] = np.nanmin(significance_bin)

        return max_significance, mean_significance, min_significance
    
    def compute_anisotropy_f(self, f_bins=np.arange(1e-4, 5e-3, 1e-4), in_db = True):
        """
        Computes the anisotropy factor f from the 2D PSD DataArray.

        Params:
        f_bins : np.Array
            Frequency bins
        in_db : Bool
            Flag indicating if the input is in dBs

        Returns:
        anisotropy : float
            Anisotropy vector, defined as the ratio of the maximum to minimum PSD values for each frequency.
        """
        # Get the radial frequency as the quadratic sum of fx and fy
        f = np.sqrt(self.psd['freq_x']**2 + self.psd['freq_y']**2)

        # Get the PSD values
        psd_values = self.psd.values
        if in_db:
            psd_values = 10 ** (psd_values / 10)

        # Bin the PSD values based on the radial frequency
        anisotropy = np.zeros_like(f_bins[:-1])
        for i, (f_min, f_max) in enumerate(zip(f_bins[:-1], f_bins[1:])):
            # Get the bin mask and mask the PSD for that bin
            mask = (f >= f_min) & (f < f_max)
            psd_bin = psd_values[mask]

            # Compute the anisotropy if the bin is not empty
            if len(psd_bin) > 0:
                anisotropy[i] = (np.std(psd_bin)) / (np.mean(psd_bin))

        return anisotropy

    def compute_phase_coherence_f(self,
                              f_bins=np.arange(1e-4, 5e-3, 1e-4),
                              theta_bins=np.deg2rad(np.arange(0, 181, 15))):
        """
        Computes phase-coherence statistics per (radial frequency, angle) bin.

        Params:
            f_bins : np.Array
                Frequency bins
            theta_bins : np.Array
                Theta bins

        Returns
        phase_coherence_max  : ndarray (n_f, n_theta)
            Phase coherence at the location of maximum PSD in the bin.
        max_phase_coherence  : ndarray (n_f, n_theta)
            Maximum phase coherence in the bin.
        min_phase_coherence  : ndarray (n_f, n_theta)
            Minimum phase coherence in the bin.
        mean_phase_coherence : ndarray (n_f, n_theta)
            Mean phase coherence in the bin.
        """
        # Get the fx and fy frequencies and obtain a meshgrid
        fx = self.psd['freq_x'].values
        fy = self.psd['freq_y'].values
        FX, FY = np.meshgrid(fx, fy, indexing='xy')

        # Get the radial frequency and polar angle theta
        F = np.sqrt(FX**2 + FY**2)
        THETA = np.mod(np.arctan2(FY, FX), np.pi)

        # Get the PSD and phase coherence values
        psd_values = self.psd.values
        coh_values = self.phase_coherence.values

        # Get the number of bins
        n_f = len(f_bins) - 1
        n_t = len(theta_bins) - 1

        # Preallocate the memory for the output variables
        phase_coherence_max  = np.full((n_f, n_t), np.nan)
        max_phase_coherence  = np.full((n_f, n_t), np.nan)
        min_phase_coherence  = np.full((n_f, n_t), np.nan)
        mean_phase_coherence = np.full((n_f, n_t), np.nan)

        # For every frequency and theta bin
        for i, (f_min, f_max) in enumerate(zip(f_bins[:-1], f_bins[1:])):
            for j, (t_min, t_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):

                # Get the bin mask
                mask = (F >= f_min) & (F < f_max) & (THETA >= t_min) & (THETA < t_max)
                if not np.any(mask):
                    continue

                # Mask the bin PSD and phase coherence
                psd_bin = psd_values[mask]
                coh_bin = coh_values[mask]

                # Compute the output statistics
                psd_max_idx = np.argmax(psd_bin)                    # Get the bin max PSD values
                phase_coherence_max[i, j]  = coh_bin[psd_max_idx]   # Coherence of the max PSD values
                max_phase_coherence[i, j]  = np.max(coh_bin)        # Max coherence of the bin
                min_phase_coherence[i, j]  = np.min(coh_bin)        # Min coherence of the bin
                mean_phase_coherence[i, j] = np.mean(coh_bin)       # Mean coherence of the bin

        return phase_coherence_max, max_phase_coherence, min_phase_coherence, mean_phase_coherence

    def compute_wavelet_2D(self,var_name='Sigma0_VV_no_targets',spatial_dims=('y', 'x'),n_octaves=6,scales_per_octave=2,n_angles=18,output_stride=16,sigma=6.0,k0=np.pi,detrend=True):
        """
        Computes a 2D directional Continuous Wavelet Transform (CWT) on the tile
        using a rotated complex Morlet wavelet. Returns the transformed image.
        """
        dim_y, dim_x = spatial_dims
        da = self.tile[var_name].transpose(dim_y, dim_x)
        vals = da.values.astype(np.float32)
        vals = np.where(np.isfinite(vals), vals, np.nanmean(vals))
        vals -= vals.mean()
        ny, nx = vals.shape

        if detrend:
            F_detrend = scipy.fft.fft2(vals, workers=-1)
            ky_d = (2 * np.pi * np.fft.fftfreq(ny))[:, None]
            kx_d = (2 * np.pi * np.fft.fftfreq(nx))[None, :]
            k_mag = np.sqrt(kx_d**2 + ky_d**2)
            k_cutoff = 2 * np.pi / 20000.0    # remove scales larger than 20 km
            F_detrend *= 1.0 - np.exp(-0.5 * (k_mag / k_cutoff)**2)
            vals = np.real(scipy.fft.ifft2(F_detrend, workers=-1)).astype(np.float32)

        # Obtain x and y sampling intervals
        dy = float(self.tile.metadata.attrs['Abstracted_Metadata:azimuth_spacing'])
        dx = float(self.tile.metadata.attrs['Abstracted_Metadata:range_spacing'])

        # Obtain spacing as the average between dx and dy
        spacing = (dx + dy) / 2
        
        # Obtain the scaling factor for the wavelets with wavelengths that go between 200 m and 10000 m. This is according to lambda = 2*pi/k * s * spacing
        s_min = 200 * k0 / (2.0 * np.pi * spacing)
        s_max = 5000 * k0 / (2.0 * np.pi * spacing)
        n_scales = n_octaves * scales_per_octave + 1
        scales = np.geomspace(s_min, s_max, n_scales)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        # Compute the FFT of the measured values
        F = scipy.fft.fft2(vals, workers=-1)
        ky = (2 * np.pi * np.fft.fftfreq(ny))[:, None].astype(np.float32)
        kx = (2 * np.pi * np.fft.fftfreq(nx))[None, :].astype(np.float32)

        # Generate the meshgrid for the obtained wavelet transform outputs
        ys = np.arange(0, ny, output_stride)
        xs = np.arange(0, nx, output_stride)
        yy_s, xx_s = np.meshgrid(ys, xs, indexing='ij')

        # Preallocate the memory for the scales, angles, and amplitudes
        n_s, n_a = len(scales), len(angles)
        amp = np.zeros((n_s, n_a, len(ys), len(xs)), dtype=np.float32)

        # Preallocate the memory for the statistics vectors
        sum_amp     = np.zeros((n_s, n_a), dtype=np.float64)
        sum_amp_sq  = np.zeros((n_s, n_a), dtype=np.float64)
        sum_phase2  = np.zeros((n_s, n_a), dtype=np.complex128)  # for Γ
        
        pixel_count = 0
        for i_s, scale in enumerate(scales):
            for i_a, theta in enumerate(angles):
                # Generate the wavenumbers parallel and perpendicular to the wavelet
                ct, st = np.cos(theta), np.sin(theta)
                k_par  =  ct * kx + st * ky
                k_perp = -st * kx + ct * ky
                k_r    = np.sqrt(k_par**2 + k_perp**2)

                # Obtain the fourier transform of the wavelet function
                psi_hat = scale * np.exp(
                    -0.5 * sigma**2 * ((scale * k_par - k0)**2 + (scale * k_perp)**2)
                )
                
                # Enforce analyticity: kill the negative-frequency lobe. Make the wavelet complex in space domain
                psi_hat = np.where(k_par > 0, psi_hat, 0.0)

                # Obtain the wavelet transform as the correlation in time between the wavelet and the signal. That's why the conjugate
                W = scipy.fft.ifft2(F * np.conj(psi_hat), workers=-1)

                # Subsample the wavelet transform since we are interested in the low frequency components. Complies with Nyquist limit
                W_sub = W[yy_s, xx_s]                    # subsample once
                A_sub = np.abs(W_sub).astype(np.float32)

                # Obtain the statistics
                amp[i_s, i_a]        = A_sub
                sum_amp[i_s, i_a]    = A_sub.mean()
                sum_amp_sq[i_s, i_a] = (A_sub ** 2).mean()

                # Phase-gradient coherence
                # 1. Get the phase of the CWT
                phi = np.angle(W_sub)

                # 2. Compute the expected carrier phase at each pixel
                carrier_phase = (k0 / scale) * (
                    xx_s * np.cos(theta) + yy_s * np.sin(theta)
                )

                # 3. Demodulate
                phi_demod = phi - carrier_phase

                # 4. Average the unit phasors and take the modulus
                sum_phase2[i_s, i_a] = np.abs(np.exp(1j * 2.0 * phi_demod).mean())

        # Obtantion of statistics
        theta_best_idx  = np.argmax(sum_amp, axis=1)                      # The index of the angle with highest amplitudes
        A_best          = sum_amp[np.arange(n_s), theta_best_idx]         # The highest amplitudes for each scale
        A_median_theta  = np.median(sum_amp, axis=1)                      # The median amplitude of all amplitudes
        anisotropy      = A_best / (A_median_theta + 1e-12)               # The anisotropy with respect to the mean value
        scale_contrast  = A_best / (np.median(A_best) + 1e-12)            # Analogy of anisotropy but at the dominant angle
        
        gamma_per_scale = np.abs(sum_phase2[np.arange(n_s), theta_best_idx])  # Measures the coherence of the obtained structure [0 (minimum) to 1 (maximum)]
                                                                                  # High coherence means that all values in the wavelet image have the same phase
        theta_deg_per_scale = np.rad2deg(angles[theta_best_idx])                 # Obtain the angle of the dominant structure at each scale
        lambda_per_scale    = 2.0 * np.pi * scales * spacing / k0                # Obtain the wavelength of all scales

        scale_best_idx = int(np.argmax(A_best))                            # Find the dominant scale

        stats = xr.Dataset(
            data_vars={
                'mean_amplitude':   (('scale', 'angle'), sum_amp),
                'A_best':           (('scale',),         A_best),
                'A_median_theta':   (('scale',),         A_median_theta),
                'anisotropy':       (('scale',),         anisotropy),
                'scale_contrast':   (('scale',),         scale_contrast),
                'gamma':            (('scale',),         gamma_per_scale),
                'theta_best_idx':   (('scale',),         theta_best_idx),
                'theta_deg':        (('scale',),         theta_deg_per_scale),
                'wavelength_m':     (('scale',),         lambda_per_scale),
            },
            coords={'scale': scales, 'angle': angles},
            attrs={
                'scale_best_idx':     scale_best_idx,
                'theta_best_deg':     float(theta_deg_per_scale[scale_best_idx]),
                'wavelength_best_m':  float(lambda_per_scale[scale_best_idx]),
                'gamma_best':         float(gamma_per_scale[scale_best_idx]),
                'anisotropy_best':    float(anisotropy[scale_best_idx]),
                'scale_contrast_best': float(scale_contrast[scale_best_idx]),
                'spacing_m':          float(spacing),
                'dx_m':               dx,
                'dy_m':               dy,
                'sigma':              sigma,
                'k0':                 k0,
                'source_var':         var_name,
            },
        )
        self.stats = stats
        return stats


    def compute_autocorr_2D(self, var_name="Sigma0_VV_no_targets", spatial_dims=("y", "x"), filter=True, normalize=True):
        """Computes 2D Autocorrelation on an xarray DataArray via FFT and returns

        a DataArray with physical spatial lag coordinates (in meters).

        Params:
        var_name : String
            Name of the variable to be processed.
        spatial_dims : Tuple of strings
            Name of the spatial dimensions of the image.
        filter : Bool
            Whether to apply a 2D Hanning window before computing autocorrelation.
        normalize : Bool
            If True, normalizes the autocorrelation peak at zero-lag to 1.0.

        Returns:
        autocorr_da : xr.DataArray
            The 2D autocorrelation surface indexed by spatial lags (lag_y, lag_x).
        """
        dim_y, dim_x = spatial_dims

        # Fill NaNs before FFT operations (FFTs cannot process NaNs)
        da_filled = self.tile[var_name].fillna(np.mean(self.tile[var_name]))

        # Subtract mean component (Crucial for autocorrelation to isolate spatial structures)
        da_demeaned = da_filled - np.mean(da_filled)

        # Spatial dimensions
        Ny, Nx = da_demeaned.shape

        # Retrieve sampling intervals (dy, dx) in meters from metadata
        dy = self.tile.metadata.attrs["Abstracted_Metadata:azimuth_spacing"]
        dx = self.tile.metadata.attrs["Abstracted_Metadata:range_spacing"]

        if filter:
            # Apply a 2D Hanning window to suppress boundary discontinuity artifacts
            win_y = np.hanning(Ny)
            win_x = np.hanning(Nx)
            window_2d = np.outer(win_y, win_x)

            da_demeaned = da_demeaned * window_2d

        # --- Wiener–Khinchin Theorem for 2D Autocorrelation ---
        # 1. Zero-pad to avoid circular correlation / aliasing effects
        pad_shape = (2 * Ny - 1, 2 * Nx - 1)

        # 2. FFT -> Power Spectral Density (PSD) -> Inverse FFT
        fft_vals = np.fft.fft2(da_demeaned, s=pad_shape)
        psd_raw = np.abs(fft_vals) ** 2
        autocorr_raw = np.fft.ifft2(psd_raw).real

        # 3. Shift zero-lag (maximum correlation peak) to center
        autocorr_shifted = np.fft.fftshift(autocorr_raw)

        # Normalize autocorrelation so zero-lag peak = 1.0
        if normalize:
            zero_lag_val = autocorr_shifted[
                autocorr_shifted.shape[0] // 2, autocorr_shifted.shape[1] // 2
            ]
            if zero_lag_val > 0:
                autocorr_out = autocorr_shifted / zero_lag_val
            else:
                autocorr_out = autocorr_shifted
            self.tile["autocorr_peak_value"] = zero_lag_val
        else:
            autocorr_out = autocorr_shifted

        # --- Construct Physical Spatial Lag Axes (in meters) ---
        # Centered lags ranging from -(N-1)*d to +(N-1)*d
        lags_y = np.arange(-(Ny - 1), Ny) * dy
        lags_x = np.arange(-(Nx - 1), Nx) * dx

        # Construct output DataArray matching the xarray workflow
        autocorr_da = xr.DataArray(
            autocorr_out,
            coords={"lag_y": lags_y, "lag_x": lags_x},
            dims=["lag_y", "lag_x"],
            name="autocorrelation",
        )
        autocorr_da.lag_y.attrs["units"] = "m"
        autocorr_da.lag_x.attrs["units"] = "m"

        # Save to object attribute and return
        self.autocorr = autocorr_da

        return autocorr_da

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

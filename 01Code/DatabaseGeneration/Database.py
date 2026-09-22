import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import datetime
import re
import numpy as np

class Database:

    def __init__(self,db_path,sar_src_path='',fino_src_path='',images_path=''):

        # Add paths to processed SAR files and FINO1
        self.sar_src_path = sar_src_path
        self.fino_src_path = fino_src_path
        
        # Add path to the database file
        self.db_path = db_path

        # Create DB directory
        dirname, fname = os.path.split(db_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        # Add path to the images folder
        self.images_path = images_path
        
        # Create images directory
        dirname, fname = os.path.split(images_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def save_db_to_pickle(self):
        self.db.to_pickle(self.db_path)

    def load_db(self):
        self.db = pd.read_pickle(self.db_path)

    def merge_datasets(self):
        
        # Load FINO1 dataset
        self.fino1_df = pd.read_pickle(self.fino_src_path)

        # Get paths of SAR files
        SAR_files = glob.glob(os.path.join(self.sar_src_path,'*.nc'))
        
        # Instantiate database dataframe
        db = pd.DataFrame()
        for file in SAR_files:
            
            # Extract the measurement time
            match = re.search(r"\d{8}T\d{6}", file)
            dt_str = match.group(0)
            date_time = datetime.datetime.strptime(dt_str,"%Y%m%dT%H%M%S")

            # Get the closest FINO1 measurement time
            row = self.get_closest_measurement(date_time)

            # Extract the filename and add it to the dataframe
            filename = os.path.basename(file)
            row['filename'] = filename
            db = pd.concat([db,row])

        db = db.reset_index(drop=True)
        # Add created db to the database
        self.db = db

        # Save the database into a pickle
        self.save_db_to_pickle()

        print(f'Database has been generated at {self.db_path}')
        return db
         
    def get_closest_measurement(self, date_time, datetime_col='TIME'):
            """
            Finds the row in FINO1 dataframe self.df where datetime_col is closest to target_time_str.
            Requires that the FINO1 dataframe is read with self.read_fino_file()
    
            Params:
            date_time: Datetime.datetime
                Datetime to find in FINO1 dataset
            datetime_col : String
                Name of the datetime column in FINO1 dataset
            
            Returns:
            self.tile : xr.Dataset
                The cropped SAR tile with the added FINO1 information corresponding to the SAR measurement time.
            """
            target_time_str = date_time

            # Parse target string into a pandas Timestamp
            target_dt = pd.to_datetime(target_time_str)
            
            # Ensure the dataframe column is datetime type
            dt_series = pd.to_datetime(self.fino1_df[datetime_col])
            
            # Find index of minimum absolute time difference
            closest_idx = (dt_series - target_dt).abs().idxmin()
            
            # Return the row as a DataFrame (or Series via df.loc[closest_idx])
            row = self.fino1_df.loc[[closest_idx]]
        
            return row

    def load_sar_product(self, idx):
        """
            Loads the pickle corresponding to the idx row of the database
        Params:
        idx : int
            Index of the SAR product of the database to be loaded
        """

        # Get the file path corresponding to the target record
        filename = self.db.iloc[idx]['filename']
        
        path = os.path.join(self.sar_src_path,filename)
        
        # Load the product
        tile = xr.open_dataset(path)
        self.tile = tile

        self.sar_product = filename
        self.sar_product_meteo = self.db.iloc[idx]

        return tile

    ######### PLOT FUNCTIONS #########
    def generate_db_images(self,images_path=''):
        """
        Generates images for the whole database with scene images and FFT images

        Params:
        images_path = String
            Path to where the images will be stored
        """

        if images_path != '':
            self.images_path = images_path

        for index in self.db.index:
            print(f'Generating image {index}')
            self.load_sar_product(index)
            self.plot_scene(save_image=True,images_path=self.images_path)
            self.plot_fft(save_image=True,images_path=self.images_path)
            self.plot_fft(zoom=True,save_image=True,images_path=self.images_path)
            self.plot_autocorr(save_image=True,images_path=self.images_path)
            self.plot_autocorr(zoom=True,save_image=True,images_path=self.images_path)
            print(f'Image saved at {self.images_path}')

    def plot_scene(self, var_name="Sigma0_VV_no_targets", clim_low=0, clim_high=0.1, normalize = True, save_image=False, images_path=''):
        """Plots a spatial map of a specified target variable from the SAR tile
        using latitude and longitude coordinates.

        Params:
        var_name : str, optional
            The variable inside self.tile to display (default:
            'Sigma0_VV_no_targets').
        clim_low : float, optional
            Minimum colorbar display threshold (default: 0).
        clim_high : float, optional
            Maximum colorbar display threshold (default: 0.1).
        """
        fig = plt.figure()
        
        # Render spatial tile using longitude/latitude coordinates
        if normalize == True:
            plot = (self.tile[var_name]/np.nanmax(self.tile[var_name])).plot(x="lon", y="lat")
        else:
            plot = self.tile[var_name].plot(x="lon", y="lat")

        # Set dynamic range / color intensity limits for backscatter values
        plot.set_clim(clim_low, clim_high)

        # Get current plot axes handle for adding annotation overlays
        ax = fig.gca()

        # Overlay Bulk Richardson Number (bRi) if available in tile metadata
        ax.text(
            0.05,
            0.92,
            f"bRi: {self.sar_product_meteo['Ri']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )

        # Overlay Wind Shear Exponent (alpha) if available (r"" used for LaTeX \alpha)
        ax.text(0.05,
            0.82,
            rf"$\alpha$: {self.sar_product_meteo['wind_shear_exponent']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )

        # Overlay Obukhov Length (L) if available in tile metadata
        ax.text(
            0.05,
            0.72,
            f"L: {self.sar_product_meteo['L']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        
        plt.title(self.sar_product)

        if save_image == True:
            img_path = os.path.join(images_path,self.sar_product.split('.')[0])
            plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        

    def plot_fft(self, clim_low=-40, clim_high=0, zoom=False, save_image=False, images_path=''):
        """Plots the 2D Power Spectral Density (PSD) calculated from the SAR imagery
        and overlays corresponding atmospheric stability metrics (Ri, alpha, L).

        Params:
        clim_low : float, optional
            Lower limit for the spectral power intensity scale in dB (default: 30).
        clim_high : float, optional
            Upper limit for the spectral power intensity scale in dB (default: 50).
        zoom : bool, optional
            If True, zooms in on low spatial frequency components near the spectral
            center.
            If False, plots the full PSD spectrum (default: False).
        """
        fig = plt.figure()

        # Slice spatial frequencies if zoom mode is enabled
        if not zoom:
            plot = self.tile.psd.plot()
        else:
            # Crop frequency axes around central low-frequency domain
            plot = self.tile.psd.sel(
                freq_x=slice(-0.005, 0.005), freq_y=slice(-0.005, 0.005)
            ).plot()

        # Apply colormap and set power intensity bounds
        plot.set_cmap("viridis")
        plot.set_clim(clim_low, clim_high)

        plt.title(self.sar_product)

        # Get current plot axes handle for adding annotation overlays
        ax = plt.gca()

        # Overlay Bulk Richardson Number (bRi) if available in tile metadata
        if "Ri" in self.tile:
            ax.text(
                0.05,
                0.92,
                f"bRi: {self.tile.Ri.item():.3f}",
                transform=ax.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )

        # Get current plot axes handle for adding annotation overlays
        ax = fig.gca()

        # Overlay Bulk Richardson Number (bRi) if available in tile metadata
        ax.text(
            0.05,
            0.92,
            f"bRi: {self.sar_product_meteo['Ri']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )

        # Overlay Wind Shear Exponent (alpha) if available (r"" used for LaTeX \alpha)
        ax.text(0.05,
            0.82,
            rf"$\alpha$: {self.sar_product_meteo['wind_shear_exponent']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )

        # Overlay Obukhov Length (L) if available in tile metadata
        ax.text(
            0.05,
            0.72,
            f"L: {self.sar_product_meteo['L']:.3f}",
            transform=ax.transAxes,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        
        plt.title(self.sar_product)

        if save_image == True:
            if zoom == False:
                img_path = os.path.join(images_path,self.sar_product.split('.')[0]+'_FFT')
                plt.savefig(img_path, dpi=150, bbox_inches="tight")
            else:
                img_path = os.path.join(images_path,self.sar_product.split('.')[0]+'_FFT_zoom')
                plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_autocorr(self, clim_low=0, clim_high=0.1, zoom=False, save_image=False, images_path=''):
            """Plots the 2D Power Spectral Density (PSD) calculated from the SAR imagery
            and overlays corresponding atmospheric stability metrics (Ri, alpha, L).
    
            Params:
            clim_low : float, optional
                Lower limit for the spectral power intensity scale in dB (default: 30).
            clim_high : float, optional
                Upper limit for the spectral power intensity scale in dB (default: 50).
            zoom : bool, optional
                If True, zooms in on low spatial frequency components near the spectral
                center.
                If False, plots the full PSD spectrum (default: False).
            """
            fig = plt.figure()
    
            # Slice spatial frequencies if zoom mode is enabled
            if not zoom:
                plot = self.tile.autocorr.plot()
            else:
                # Crop frequency axes around central low-frequency domain
                plot = self.tile.autocorr.sel(
                    lag_x=slice(-10e3, 10e3), lag_y=slice(-10e3, 10e3)
                ).plot()
    
            # Apply colormap and set power intensity bounds
            plot.set_cmap("viridis")
            plot.set_clim(clim_low, clim_high)
    
            plt.title(self.sar_product)
    
            # Get current plot axes handle for adding annotation overlays
            ax = plt.gca()
    
            # Overlay Bulk Richardson Number (bRi) if available in tile metadata
            if "Ri" in self.tile:
                ax.text(
                    0.05,
                    0.92,
                    f"bRi: {self.tile.Ri.item():.3f}",
                    transform=ax.transAxes,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6),
                )
    
            # Get current plot axes handle for adding annotation overlays
            ax = fig.gca()
    
            # Overlay Bulk Richardson Number (bRi) if available in tile metadata
            ax.text(
                0.05,
                0.92,
                f"bRi: {self.sar_product_meteo['Ri']:.3f}",
                transform=ax.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )
    
            # Overlay Wind Shear Exponent (alpha) if available (r"" used for LaTeX \alpha)
            ax.text(0.05,
                0.82,
                rf"$\alpha$: {self.sar_product_meteo['wind_shear_exponent']:.3f}",
                transform=ax.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )
    
            # Overlay Obukhov Length (L) if available in tile metadata
            ax.text(
                0.05,
                0.72,
                f"L: {self.sar_product_meteo['L']:.3f}",
                transform=ax.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )
            
            plt.title(self.sar_product)
    
            if save_image == True:
                if zoom == False:
                    img_path = os.path.join(images_path,self.sar_product.split('.')[0]+'_AC')
                    plt.savefig(img_path, dpi=150, bbox_inches="tight")
                else:
                    img_path = os.path.join(images_path,self.sar_product.split('.')[0]+'_AC_zoom')
                    plt.savefig(img_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        
    def plot_wind_shear_profile(self):
        # Plot the wind shear fit for the first wind profile
        plt.figure()
        # Plot the actual wind profile
        wind_speed_columns = [col for col in self.sar_product_meteo.columns if col.startswith('WSPD') and 'CUP' in col and "MAX" not in col and "MIN" not in col and "MC" in col and "VAR" not in col]
        altitudes = [int(col.split('_')[-1][:-1]) for col in wind_speed_columns]  # Extract altitudes from column names

        wind_speeds = self.sar_product_meteo[wind_speed_columns].values  # Extract wind speed values 

        idx = 5
        # Sort altitudes and corresponding wind speeds
        sorted_indices = np.argsort(altitudes)
        altitudes = np.array(altitudes)[sorted_indices]
        wind_speeds = wind_speeds[:, sorted_indices]
        plt.plot(wind_speeds[idx,:], altitudes)

        # Plot the fitted wind shear line
        z1 = altitudes[0]
        z2 = altitudes[-1]
        WSPD_z1 = wind_speeds[0]
        WSPD_z2 = wind_speeds[-1]
        alpha = self.sar_product_meteo['wind_shear_exponent'].iloc[idx]

        fitted_profile = WSPD_z1*(altitudes/z1)**alpha
        plt.plot(fitted_profile,altitudes)   
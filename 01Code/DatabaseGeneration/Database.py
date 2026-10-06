import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import datetime
import re
import numpy as np
import traceback

class Database:

    def __init__(self,db_path,sar_src_path='',sar_statistics_path='',fino_src_path='',images_path=''):

        # Add paths to processed SAR files and FINO1
        self.sar_src_path = sar_src_path
        self.sar_statistics_path = sar_statistics_path
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

    def merge_datasets(self,statistics=True):
        
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

            if statistics == True:
                try:
                    # Load the SAR statistics file
                    stats_file = os.path.join(self.sar_statistics_path,filename.split('.')[0]+'_statistics.nc')
                    self.stats = xr.open_dataset(stats_file)

                    # Extract the statistics and add them to the row
                    row['max_component'] = [self.stats['max_component'].values]
                    row['max_component_coherence'] = [self.stats['max_component_coherence'].values]
                    row['anisotropy'] = [self.stats['anisotropy'].values]
                    row['phase_coherence_max'] = [self.stats['phase_coherence_max'].values]
                    row['max_phase_coherence'] = [self.stats['max_phase_coherence'].values]
                    row['min_phase_coherence'] = [self.stats['min_phase_coherence'].values]
                    row['mean_phase_coherence'] = [self.stats['mean_phase_coherence'].values]
                    row['f_bins'] = [self.stats['f_bins'].values]
                except Exception as e:
                    print(f'Could not add statistics')
                    print(f'Error: {type(e).__name__}: {e}')
                    traceback.print_exc()
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
            try:
                self.load_sar_product(index)
                self.plot_scene(save_image=True,images_path=self.images_path)
                self.plot_fft(save_image=True,images_path=self.images_path)
                self.plot_fft(zoom=True,save_image=True,images_path=self.images_path)
                self.plot_autocorr(save_image=True,images_path=self.images_path)
                self.plot_autocorr(zoom=True,save_image=True,images_path=self.images_path)
            except:
                print('Some printing failed')
            print(f'Image saved at {self.images_path}')

    def generate_db_images_spectral_analysis(self,product_index,images_path=''):
        
        if images_path != '':
            self.images_path = images_path
        else:
            images_path=self.images_path
        
        print(f'Generating image {product_index}')
        try:
            self.load_sar_product(product_index)
            
            fname = self.sar_product.split('.')[0]

            psd = self.tile.psd.values
            ncrs = self.tile['Sigma0_VV_no_targets'].values
            lat = self.tile.lat.values
            lon = self.tile.lon.values
            f_x = self.tile.freq_x.values
            f_y = self.tile.freq_y.values
            phase_coherence = self.tile.phase_coherence.values
            max_component = self.tile.max_component.values
            max_component_coherence = self.tile.max_component_coherence.values
            anisotropy = self.tile.anisotropy.values
            phase_coherence_max = self.tile.phase_coherence_max.values
            max_phase_coherence = self.tile.max_phase_coherence.values
            min_phase_coherence = self.tile.min_phase_coherence.values
            mean_phase_coherence = self.tile.mean_phase_coherence.values
            f_bins = self.tile.f_bins.values
            # Convert PSD from dB to linear
            psd = 10.0 ** (psd / 10.0)

            # Drop the last frequency bin (MATLAB 1-indexed: f_bins(1:end-1))
            f_bins = f_bins[:-1]

            # Build masks for |f| < 5e-2
            idx_x = np.abs(f_x) < 5e-2
            idx_y = np.abs(f_y) < 5e-2

            # Subset (numpy uses [row, col] = [y, x])
            psd = psd[np.ix_(idx_y, idx_x)]
            phase_coherence = phase_coherence[np.ix_(idx_y, idx_x)]
            f_x_sub = f_x[idx_x]
            f_y_sub = f_y[idx_y]

            # --- Build figure (30 x 20 cm) ---
            cm_to_in = 1 / 2.54
            fig = plt.figure(figsize=(30 * cm_to_in, 20 * cm_to_in), constrained_layout=True)

            # --- Subplot 1: PSD ---
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3), useMathText=True)
            psd_db = 10 * np.log10(psd)
            # pcolormesh expects edges; use centers and shading='auto'
            mesh1 = ax1.pcolormesh(f_x_sub, f_y_sub, psd_db, shading='auto')
            ax1.set_xlabel('f_y [1/m]')
            ax1.set_ylabel('f_x [1/m]')
            ax1.set_xlim([-1e-3, 1e-3])
            ax1.set_ylim([-1e-3, 1e-3])
            ax1.set_title(fname, fontsize=8)
            fig.colorbar(mesh1, ax=ax1)

            # --- Subplot 2: Phase coherence ---
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.ticklabel_format(axis='both', style='sci', scilimits=(-3, -3), useMathText=True)
            pc_db = 10 * np.log10(phase_coherence)
            mesh2 = ax2.pcolormesh(f_x_sub, f_y_sub, pc_db, shading='auto',
                                vmin=-10, vmax=0)
            ax2.set_xlabel('f_y [1/m]')
            ax2.set_ylabel('f_x [1/m]')
            ax2.set_xlim([-1e-3, 1e-3])
            ax2.set_ylim([-1e-3, 1e-3])
            fig.colorbar(mesh2, ax=ax2)

            # --- Subplot 3: Max component vs wavelength ---
            ax3 = fig.add_subplot(2, 3, 3)
            wavelength = 1.0 / f_bins
            ax3.plot(wavelength, max_component)
            ax3.plot(wavelength, max_component_coherence, label='Max Component Coherence', linestyle='--', color='r')
            ax3.set_xlabel('wavelength')
            ax3.set_ylabel('PSD max component [dB normalized]')
            ax3.set_ylim([-20, 0])
            ax3.grid(True, which='both', axis='both', alpha=0.4, linestyle='--', linewidth=0.5)

            ax3 = fig.add_subplot(2, 3, 4)
            # Render spatial tile using longitude/latitude coordinates
            mesh3 = ax3.pcolormesh(lon, lat, ncrs, shading='auto', vmin=0, vmax=0.1)
            ax3.set_xlabel('Lon [deg]')
            ax3.set_ylabel('Lat [deg]')
            fig.colorbar(mesh3, ax=ax3)
    
            # Overlay Bulk Richardson Number (bRi) if available in tile metadata
            ax3.text(
                0.05,
                0.92,
                f"bRi: {self.sar_product_meteo['Ri']:.3f}",
                transform=ax3.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )
    
            # Overlay Wind Shear Exponent (alpha) if available (r"" used for LaTeX \alpha)
            ax3.text(0.05,
                0.82,
                rf"$\alpha$: {self.sar_product_meteo['wind_shear_exponent']:.3f}",
                transform=ax3.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )
    
            # Overlay Obukhov Length (L) if available in tile metadata
            ax3.text(
                0.05,
                0.72,
                f"L: {self.sar_product_meteo['L']:.3f}",
                transform=ax3.transAxes,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6),
            )

            # --- Subplot 5: Phase coherence max vs wavelength ---
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.plot(wavelength, phase_coherence_max, color='k')
            ax5.plot(wavelength, max_phase_coherence, label='Max Phase Coherence', linestyle='--', color='r')
            ax5.plot(wavelength, min_phase_coherence, label='Min Phase Coherence', linestyle='--', color='r')
            ax5.plot(wavelength, mean_phase_coherence, label='Mean Phase Coherence', linestyle='-.', color=[0.5, 0.5, 0.5])
            ax5.set_xlabel('wavelength')
            ax5.set_ylabel('Gamma')
            ax5.set_ylim([0, 1])
            # Dashed reference line at 0.1
            ax5.axhline(y=0.1, linestyle='--', color='k')
            ax5.grid(True, which='both', axis='both', alpha=0.4, linestyle='--', linewidth=0.5)

            # --- Subplot 6: Anisotropy vs wavelength ---
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.plot(wavelength, anisotropy)
            ax6.set_xlabel('Wavelength')
            ax6.set_ylabel('Anisotropy')
            ax6.grid(True, which='both', axis='both', alpha=0.4, linestyle='--', linewidth=0.5)
            ax6.set_ylim([0, 3])

            # --- Save ---
            base = fname[:-3] if fname.endswith('.nc') else fname
            out_path = os.path.join(images_path, f'{base}_analysis.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Image saved at {out_path}')

        except Exception as e:
            print(f'Could not process file')
            print(f'Error: {type(e).__name__}: {e}')
            traceback.print_exc()

    def generate_db_images_for_single_product(self,product_index,images_path='',scene=True,fft=True,AC=True):
            """
            Generates images for the whole database with scene images and FFT images
    
            Params:
            images_path = String
                Path to where the images will be stored
            """
    
            if images_path != '':
                self.images_path = images_path
    
            print(f'Generating image {product_index}')
            try:
                self.load_sar_product(product_index)
                if scene == True:
                    self.plot_scene(save_image=True,images_path=self.images_path)

                if fft == True:
                    self.plot_fft(save_image=True,images_path=self.images_path)
                    self.plot_fft(zoom=True,save_image=True,images_path=self.images_path)

                if AC == True:
                    self.plot_autocorr(save_image=True,images_path=self.images_path)
                    self.plot_autocorr(zoom=True,save_image=True,images_path=self.images_path)
            except:
                print('Some printing failed')
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
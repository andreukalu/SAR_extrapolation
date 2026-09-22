# IMPORTS
import re
import pandas as pd
import netCDF4 as nc
import numpy as np
import os
import glob
import AtmosphericProcessor

# File Processor Class
"""
    This class takes as input all the available .nc files in fino_src_path, pre-process them, and merges them into
    a single pd.Dataframe. 
    All .nc files should follow FINO1 file convention. Preferably they all should be FINO1 measurement files.
    
"""
class FINO1Processor:
    
    ######### Constructor ##############################
    def __init__(self, fino_src_path, fino_dst_path):
        """
            Class constructor. Definition of the folder paths.
        Params:
        fino_src_path : String
            Path to the source folder where all FINO1 .nc files are stored
        fino_dst_path : String
            Path to the destination folder where the processed dataframes will be stored
        """
        self.fino_src_path = fino_src_path  # Source path for input files
        self.fino_dst_path = fino_dst_path  # Destination path for output files
        
        # Create FINO dst directory
        os.makedirs(self.fino_dst_path, exist_ok=True)

    ######### Methods ##################################
    def process_fino_files(self):
        """
            Read and process all the FINO1 files, generate a unified dataframe, compute the 2nd order atmospheric parameters,
            and save the file as a pickle
        """
        # Read all .nc FINO1 files in fino_src_path
        self.read_files()

        # Compute the 2nd order atmospheric parameters
        self.compute_atmospheric_parameters()

        # Save the processed dataframe as a pickle in fino_dst_folder
        self.write_pickle()

    def read_files(self):
        """
            Read all the files in fino_src_path
        """

        # Get all the available .nc files in self.fino_src_path
        files = glob.glob(os.path.join(self.fino_src_path,'*.nc'))

        # Read and process each file in self.fino_src_path
        for file in files:
            self.read_netcdf(file)

    def read_netcdf(self, file_name):
        """
        Read a NetCDF file and return a DataFrame containing the data as a dataframe.
        
        Params:
        file_name : String
            File name of the .nc file to be read.
        
        Returns:
        data_df : pd.Dataframe
            Dataframe containing the formatted data in the input file. 
            If previous files were already processed, the new dataframe is concatenated to the preprocessed ones.
        """
        file_path = os.path.join(self.fino_src_path, file_name)
        print(f"Reading NetCDF file: {file_path}")

        with nc.Dataset(file_path, 'r') as dataset:
            time_vals = dataset.variables['TIME'][:]
            depth_vals = dataset.variables['DEPTH'][:]
            
            df_list = []
            
            for var in dataset.variables:
                # Skip Quality Control (QC) variables
                if 'QC' in var.upper():
                    continue

                var_obj = dataset.variables[var]
                
                # Target 4D measurement variables (TIME, DEPTH, LON, LAT)
                if var_obj.ndim == 4:
                    # Remove text after deg (e.g., '_10deg') using regex
                    clean_var = re.sub(r'_\d+deg', '', var)
                    # Remove text after dots (e.g., '.Cup Anemometer')
                    clean_var = re.sub(r'\..*', '', var)
                    # Remove ALL numbers
                    clean_var = re.sub(r'\d+', '', clean_var)
                    # Convert USA to SONIC
                    clean_var = re.sub(r'USA+', 'SONIC', clean_var).strip('_')
                    # Clear all deg text
                    clean_var = re.sub(r'deg+', '', clean_var).strip('_')
                    # Clean up leftover underscores
                    clean_var = re.sub(r'__+', '_', clean_var).strip('_')

                    data = var_obj[:]
                    # Remove singleton dimensions (LON, LAT) -> shape (261719, 17)
                    squeezed_data = np.squeeze(data)
                    
                    # Convert MaskedArray mask to NaN
                    if np.ma.is_masked(squeezed_data):
                        squeezed_data = squeezed_data.filled(np.nan)
                    
                    # Create DataFrame for current variable
                    col_names = [f"{clean_var}_{int(d)}m" for d in depth_vals]
                    var_df = pd.DataFrame(squeezed_data, index=time_vals, columns=col_names)
                    
                    # Drop columns (depths) that contain only NaN values
                    var_df = var_df.dropna(axis=1, how='all')
                    
                    if not var_df.empty:
                        df_list.append(var_df)

            # Fix the time column
            if df_list:
                data_df = pd.concat(df_list, axis=1)
                data_df.index.name = 'TIME'
                data_df = data_df.dropna(axis=0, how='all')
            else:
                data_df = pd.DataFrame()

            # Set the time index as column and change the type to datetime
            data_df = data_df.reset_index()

            anchor_date = pd.Timestamp("2016-01-01")
            first_value = data_df["TIME"].iloc[0]

            # Subtract the starting offset to get relative elapsed days, then convert to Timedelta
            data_df["TIME"] = anchor_date + pd.to_timedelta(
                data_df["TIME"] - first_value, unit="D"
            )

            # Round to nearest minute to clean up floating point imprecision
            data_df["TIME"] = data_df["TIME"].dt.round("min")

            # Check if temperature at 0 m by the metmast is valid, and remove it if not
            if data_df['TEMP_0m'].mean() < -20:
                data_df = data_df.drop(columns=['TEMP_0m'])

            # Check if a dataframe is existent and update it with the new one
            if getattr(self, "df", None) is not None:
                if self.df.shape[0] >= data_df.shape[0]:
                    self.df = pd.merge_asof(self.df, data_df, on='TIME', direction='backward')
                else:
                    self.df = pd.merge_asof(data_df, self.df, on='TIME', direction='backward')
            else:
                self.df = data_df

        return data_df

    def read_netcdf_variable(self, file_name, variable_name):
        """
        Read a specific variable from a NetCDF file.

        Parameters:
        file_name : str
            Name of the NetCDF file to read.
        variable_name : str
            Name of the variable to extract from the NetCDF file.

        Returns:
        variable_data : np.ndarray
            Numpy array containing the data for the specified variable.
        """
        print(f"Reading variable '{variable_name}' from NetCDF file: {self.fino_src_path}/{file_name}")
        with nc.Dataset(f"{os.path.join(self.fino_src_path, file_name)}", 'r') as dataset:
            variable_data = dataset.variables[variable_name][:]
        return variable_data

    def print_info(self, file_name):
        """
        Get the variable names and their dimensions from a NetCDF file.

        Parameters:
        file_name : str
            Name of the NetCDF file to read.

        Returns:
        var_info : dict
            Dictionary containing variable names as keys and their dimensions as values.
        """
        print(f"Getting variable names and dimensions from NetCDF file: {self.fino_src_path}/{file_name}")
        with nc.Dataset(f"{os.path.join(self.fino_src_path, file_name)}", 'r') as dataset:
            var_info = {var: (dataset.variables[var].dimensions, dataset.variables[var].shape) for var in dataset.variables}
        print(var_info)

    def compute_atmospheric_parameters(self,z2=34,z1=0):
        """
        Function to compute the 2nd order atmospheric parameters 'Ri', 'wind_shear_exponent', and 'L'
        and add them to each row of the pre-processed FINO1 dataframe.

        Params:
        z2 : float
            High measurement altitude for parameter computation (m)
        z1 : float
            Low measurement altitudes for parameter computation (m)

        Returns:
        self.df : pd.Dataframe 
            Modified dataframe with the 2nd order parameters.
        """
        # Instantiate the Atmospheric processor with the pre-processed FINO1 dataframe to compute
        # the second order atmospheric parameters
        ap = AtmosphericProcessor.AtmosphericProcessor(self.df,z2,z1)

        # Compute the 2nd order parameters
        ap.compute_bulk_Richardson_number()
        ap.compute_wind_shear_exponent()
        ap.compute_Obukhov_length()

        self.df = ap.internal_df

        return self.df

    def write_pickle(self):
        """
        Write the processed FINO1 DataFrame to a pickle file.
        """
        print(f"Writing pickle file: {self.fino_dst_path}")
        self.df.to_pickle(f"{self.fino_dst_path}")
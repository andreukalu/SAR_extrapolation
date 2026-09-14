# IMPORTS
import re
import pandas as pd
import netCDF4 as nc
import numpy as np
import os

# File Processor Class
class FileProcessor:
    
    ######### Constructor ##############################
    def __init__(self, src_path, dst_path):
        self.src_path = src_path  # Source path for input files
        self.dst_path = dst_path  # Destination path for output files

    ######### Methods ##################################
    def read_netcdf(self, file_name):
        """
        Read a NetCDF file and return a DataFrame containing the data."""
        file_path = os.path.join(self.src_path, file_name)
        print(f"Reading NetCDF file: {file_path}")

        with nc.Dataset(file_path, 'r') as dataset:
            time_vals = dataset.variables['TIME'][:]
            depth_vals = dataset.variables['DEPTH'][:]
            
            df_list = []
            
            for var in dataset.variables:
                # 1. Skip Quality Control (QC) variables
                if 'QC' in var.upper():
                    continue

                var_obj = dataset.variables[var]
                
                # Target 4D measurement variables (TIME, DEPTH, LON, LAT)
                if var_obj.ndim == 4:
                    # 2. Remove text after deg (e.g., '_10deg') using regex
                    clean_var = re.sub(r'_\d+deg', '', var)
                    # 3. Remove text after dots (e.g., '.Cup Anemometer')
                    clean_var = re.sub(r'\..*', '', var)
                    # 4. Remove ALL numbers
                    clean_var = re.sub(r'\d+', '', clean_var)
                    # 6. Clean up CUP or USA text
                    # clean_var = re.sub(r'CUP+', '', clean_var).strip('_')
                    clean_var = re.sub(r'USA+', 'SONIC', clean_var).strip('_')
                    clean_var = re.sub(r'deg+', '', clean_var).strip('_')
                    # 5. Clean up leftover underscores
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

            if df_list:
                data_df = pd.concat(df_list, axis=1)
                data_df.index.name = 'TIME'
                data_df = data_df.dropna(axis=0, how='all')
            else:
                data_df = pd.DataFrame()

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
        print(f"Reading variable '{variable_name}' from NetCDF file: {self.src_path}/{file_name}")
        with nc.Dataset(f"{os.path.join(self.src_path, file_name)}", 'r') as dataset:
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
        print(f"Getting variable names and dimensions from NetCDF file: {self.src_path}/{file_name}")
        with nc.Dataset(f"{os.path.join(self.src_path, file_name)}", 'r') as dataset:
            var_info = {var: (dataset.variables[var].dimensions, dataset.variables[var].shape) for var in dataset.variables}
        print(var_info)

    def write_pickle(self, data_df, file_name):
        """
        Write a DataFrame to a pickle file.

        Parameters:
        data_df : pd.DataFrame
            DataFrame to be written to a pickle file.
        file_name : str
            Name of the output pickle file.
        """
        print(f"Writing pickle file: {self.dst_path}/{file_name}")
        data_df.to_pickle(f"{os.path.join(self.dst_path, file_name)}")
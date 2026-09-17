# IMPORTS
import numpy as np
import scipy

# Atmospheric Processing Class
class AtmosphericProcessor:

    ######### Constructor ##############################
    def __init__(self,df,z2,z1 = 0,anemometer_type='CUP'):
        self.R = 287.05  # Gas constant of air [J/(kg*K)]
        self.g = 9.81  # Gravitational acceleration [m/s^2]
        self.Cp = 1004.0  # Specific heat at constant pressure [J/(kg*K)]
        self.anemometer_type = anemometer_type # Specify the anemometer type
        self.z1 = z1  # Measurement Lower altitude (m). It should be 0
        self.z2 = z2  # Measurement Upper altitude (m)
        self.df = df  # DataFrame containing atmospheric data

        self.internal_df = self.df.copy()  # Internal DataFrame for processing

        # Get closest available altitudes to z2 for each variable
        self.z2_closest, self.z2_var_name = self.get_closest_altitude('WSPD', self.z2)
        
        # Interpolate variables between the two altitudes z1 and z2
        if self.z1 == 0: # Special case when z1 is 0, we set the values at z1 to 0 for wind speed and 98% for relative humidity
            self.internal_df["WSPD_z1"] = 0.0  # Set wind speed at sea level to 0 m/s
            self.internal_df["RELH_z1"] = 98.0  # Set relative humidity at sea level to 98%
        else:
            self.internal_df[["WSPD_z1"]] = self.interpolate_variable('WSPD', self.z1)
            self.internal_df[["RELH_z1"]] = self.interpolate_variable('RELH', self.z1)

        self.internal_df["WSPD_z2"] = self.internal_df[self.z2_var_name]  # Set wind speed at z2 to the closest available value
        self.internal_df["RELH_z2"] = self.interpolate_variable('RELH', self.z2_closest)
        self.internal_df[["TEMP_z1","TEMP_z2"]] = self.interpolate_variable('DRYT', [self.z1, self.z2_closest])
        self.internal_df[["ATMP_z1","ATMP_z2"]] = self.interpolate_variable('ATMP', [self.z1, self.z2_closest])
        
        self.convert_to_kelvin()  # Convert temperature to Kelvin
        self.convert_to_pascal()  # Convert pressure to Pascal
        
    ######### Signal Processing Methods ############################
    def get_closest_altitude(self, var_name, target_altitude):
        """
        Get the closest available altitude to the target altitude.

        Parameters:
        target_altitude : float
            The target altitude (m) for which to find the closest available altitude.

        Returns:
        closest_altitude : float
            The closest available altitude (m) to the target altitude.
        """
        # Extract the available altitudes from the DataFrame columns
        available_altitudes = [int(col.split('_')[-1][:-1]) for col in self.df.columns if col.startswith(var_name)]

        # Find the closest available altitude to the target altitude
        closest_altitude = min(available_altitudes, key=lambda x: abs(x - target_altitude))
        
        # Get the variable name corresponding to the closest altitude
        if var_name == 'WSPD':
            closest_var_name = f"{var_name}_{self.anemometer_type}_{closest_altitude}m"
        else:
            closest_var_name = f"{var_name}_{closest_altitude}m"
        return closest_altitude, closest_var_name

    def interpolate_variable(self, var_name, z):
        """
        Interpolate a variable between two altitudes.

        Parameters:
        var_name : str
            Name of the variable to interpolate (e.g., 'TEMP', 'RH', 'PRES', 'WSPD')
        z : float
            Target altitudes (m)

        Returns:
        interpolated_values : np.ndarray
            Interpolated values of the variable at the specified altitudes.
        """
        # Extract the variables data from the DataFrame
        columns = self.df.columns

        # Select columns corresponding to the variable name and altitudes
        if var_name == 'WSPD':
            selected_columns = [col for col in columns if col.startswith(var_name) and "MC" in col and self.anemometer_type in col and "MAX" not in col and "MIN" not in col and "VAR" not in col]
        else:
            selected_columns = [col for col in columns if col.startswith(var_name) and "MAX" not in col and "MIN" not in col and "MC" not in col and "VAR" not in col]

            # Sort the selected columns based on altitude and get the corresponding altitudes
            sorted_columns = sorted(selected_columns, key=lambda x: int(x.split('_')[-1][:-1]))  # Sort by altitude
        altitudes = [int(col.split('_')[-1][:-1]) for col in sorted_columns]
        var_data = self.df[sorted_columns].values
        
        # Perform linear interpolation between the two altitudes
        interpolated_values = scipy.interpolate.interp1d(altitudes, var_data, axis=1, bounds_error=False, fill_value="extrapolate")(z)

        return interpolated_values

    ######### Atmospheric Methods ##################################
    def convert_to_kelvin(self):
            """
            Convert temperature from Celsius to Kelvin in the internal DataFrame.
            """
            # Check for all temperature columns and convert them to Kelvin
            temp_columns = [col for col in self.internal_df.columns if col.startswith('TEMP') or col.startswith('DRYT')]
            for col in temp_columns:
                aux_col = self.internal_df[col].copy()
                aux_col = aux_col[(aux_col > -273.15) & (aux_col < 400)]  # Filter out invalid temperature values
                
                if aux_col.mean() < 200:  # Assuming temperatures in Celsius are less than 100
                    self.internal_df[col] = self.internal_df[col] + 273.15  # Convert to Kelvin
    
    def convert_to_pascal(self):
        """
        Convert pressure from hPa to Pascal in the internal DataFrame.
        """
        # Check for all pressure columns and convert them to Pascal
        pressure_columns = [col for col in self.internal_df.columns if col.startswith('ATMP')]
        for col in pressure_columns:
            self.internal_df[col] = self.internal_df[col] * 100  # Convert to Pascal

    def compute_bulk_Richardson_number(self):
        """
        Compute the bulk Richardson number between two altitudes.

        Returns:
        Ri : np.ndarray
            Bulk Richardson number for each time step.
        """

        print(f"Computing bulk Richardson number between {self.z1}m and {self.z2}m...")

        # Extract the necessary variables from the internal DataFrame
        T1 = self.internal_df[f"TEMP_z1"].values
        RH1 = self.internal_df[f"RELH_z1"].values
        P1 = self.internal_df[f"ATMP_z1"].values
        U1 = self.internal_df[f"WSPD_z1"].values

        T2 = self.internal_df[f"TEMP_z2"].values
        RH2 = self.internal_df[f"RELH_z2"].values
        P2 = self.internal_df[f"ATMP_z2"].values
        U2 = self.internal_df[f"WSPD_z2"].values

        # Calculate the bulk Richardson number for each time step
        Ri = np.array([self.bulk_Richardson_number(T1[i], RH1[i], T2[i], RH2[i], P1[i], P2[i], U1[i], U2[i]) for i in range(len(T1))])

        # Add the bulk Richardson number to the internal DataFrame
        self.internal_df["Ri"] = Ri

        return self.internal_df

    def bulk_Richardson_number(self, T1, RH1, T2, RH2, P1, P2, U1, U2, delta_U_lim = 2):
        """
        Calculate the bulk Richardson number between two altitudes.

        Parameters:
        T1 : float
            Temperature at lower altitude (K)
        RH1 : float
            Relative humidity at lower altitude (%)
        T2 : float
            Temperature at upper altitude (K)
        RH2 : float
            Relative humidity at upper altitude (%)
        P1 : float
            Pressure at lower altitude (Pa)
        P2 : float
            Pressure at upper altitude (Pa)
        U1 : float
            Wind speed at lower altitude (m/s)
        U2 : float
            Wind speed at upper altitude (m/s)

        Returns:
        Ri : float
            Bulk Richardson number
        """
        theta1 = self.virtual_potential_temperature(T1, RH1, P1)  # Virtual potential temperature at lower altitude
        theta2 = self.virtual_potential_temperature(T2, RH2, P2)  # Virtual potential temperature at upper altitude
        delta_theta = theta2 - theta1  # Virtual potential temperature difference
        delta_u = U2 - U1  # Wind speed difference
        if delta_u < delta_U_lim:
            return np.nan
        delta_z = self.z2 - self.z1  # Altitude difference
        mean_theta = (theta1 + theta2) / 2  # Mean virtual potential temperature

        # Calculate the bulk Richardson number
        Ri = (self.g / mean_theta) * (delta_theta / delta_z) / ((delta_u / delta_z) ** 2)
        
        return Ri

    def compute_wind_shear_exponent(self):
        """
            Calculate wind shear exponent by fitting an exponential curve to the wind speed profile between available altitudes.
        """

        print(f"Computing wind shear exponent between {self.z1}m and {self.z2}m...")

        # Extract all columns corresponding to wind speed at different altitudes
        wind_speed_columns = [col for col in self.internal_df.columns if col.startswith('WSPD') and "MC" in col and self.anemometer_type in col and "MAX" not in col and "MIN" not in col and "VAR" not in col and "z1" not in col and "z2" not in col]
        print(wind_speed_columns)
        altitudes = [int(col.split('_')[-1][:-1]) for col in wind_speed_columns]  # Extract altitudes from column names
        wind_speeds = self.internal_df[wind_speed_columns].values  # Extract wind speed values 

        # Sort altitudes and corresponding wind speeds
        sorted_indices = np.argsort(altitudes)
        altitudes = np.array(altitudes)[sorted_indices]
        wind_speeds = wind_speeds[:, sorted_indices]

        # Get reference values for the lowest altitude column
        z_ref = altitudes[0]
        u_ref = wind_speeds[:, [0]]  # Keep 2D shape (N, 1)
        
        # 3. Mask out invalid values (NaN, zero, or negative wind speeds)
        valid_mask = ~np.isnan(wind_speeds) & (wind_speeds > 0) & ~np.isnan(u_ref) & (u_ref > 0)

        # Convert power law to linear model: u(z) = u_ref * (z / z_ref) ** alpha => log(u(z)/u_ref) = alpha * log(z/z_ref)
        x = np.log(altitudes / z_ref)  # Logarithmic scale of altitudes
        y = np.log(wind_speeds / u_ref)  # Logarithmic

        # Tile 1-D x-coordinates into a 2D matrix maching the shape of y for linear regression
        x_tiled = np.tile(x, (wind_speeds.shape[0], 1))  # Shape (N, M) where N is number of time steps and M is number of altitudes

        # Closed form least-squares calculation for linear regression to find the wind shear exponent (slope)
        x_masked = np.where(valid_mask, x_tiled, 0.0)
        y_masked = np.where(valid_mask, y, 0.0)

        # Now sums ignore NaNs cleanly
        numerator = np.sum(x_masked * y_masked, axis=1)
        denominator = np.sum(x_masked**2, axis=1)

        # Require at least 2 valid height points to calculate a meaningful fit
        valid_points_count = np.sum(valid_mask, axis=1)
        sufficient_data = (denominator > 0) & (valid_points_count >= 2)

        # 6. Compute alpha and assign NaN where fit fails
        wind_shear_exponents = np.divide(
            numerator, 
            denominator, 
            out=np.full(wind_speeds.shape[0], np.nan), 
            where=sufficient_data
        )
        
        self.internal_df[f"wind_shear_exponent"] = wind_shear_exponents
        
        return self.internal_df

    def compute_Obukhov_length(self):
        """
        Compute the Obukhov length.

        Returns:
        L : np.ndarray
            Obukhov length for each time step.
        """

        print(f"Computing Obukhov length...")

        # Extract the necessary variables from the internal DataFrame
        Ri = self.internal_df["Ri"].values

        # Calculate the bulk Richardson number for each time step
        L = np.array([self.obukhov_length(Ri[i], self.z2, self.z1) for i in range(len(Ri))])

        # Add the Obukhov length to the internal DataFrame
        self.internal_df["L"] = L

        return self.internal_df
    
    def obukhov_length(self, Ri, z2, z1):
        """
        Function to compute the Obukhov length from the richardson number
        """

        # Reference height (geometric mean is often preferred, but arithmetic mean works well)
        z_p = (z2 + z1) / 2.0
        
        if Ri <= 0:
            eta = 10*Ri
        elif (Ri > 0) and (Ri < 0.2):
            eta = 10*Ri/(1-5*Ri)
        else:
            eta = np.nan

        # Compute the Obukhov length L at the reference height
        return z_p / eta if not np.isnan(eta) else np.nan

    def virtual_potential_temperature(self, T, RH, P):
        """
        Calculate Virtual Potential Temperature theta_v (K) using array inputs.
        T in Kelvin, RH in %, P in Pa.
        """
        # Mixing ratio r (kg/kg)
        r = self.mixing_ratio(T, RH, P)
        
        # Dry potential temperature theta
        theta = T * (100000.0 / P) ** (self.R / self.Cp)
        
        # Virtual potential temperature theta_v
        theta_v = theta * (1.0 + 0.61 * r)
        return theta_v

    def mixing_ratio(self, T, RH, P):
        """
        Calculate the mixing ratio.

        Parameters:
        T : float
            Temperature (K)
        RH : float
            Relative humidity (%)
        P : float
            Total atmospheric pressure (Pa)
        """
        e = self.water_vapor_pressure(T, RH)
        # Clamp (P - e) to avoid zero/negative pressure edge cases
        r =  0.622 * e / np.maximum(P - e, 1.0)
        return r

    def water_vapor_pressure(self, T, RH):
        """
        Calculate the water vapor pressure.

        Parameters:
        T : float
            Temperature (K)
        RH : float
            Relative humidity (%)

        Returns:
        e : float
            Water vapor pressure (Pa)
        """
        T_C = T - 273.15
        e_s = 611.2 * np.exp((17.67 * T_C) / (T_C + 243.5))  # Saturation pressure in Pa
        e = (RH / 100.0) * e_s

        return e
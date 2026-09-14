# IMPORTS
import numpy as np

# Atmospheric Processing Class
class AtmosphericProcessor:

    ######### Constructor ##############################
    def __init__(self,z1,z2):
        self.R = 287.05  # Gas constant of air [J/(kg*K)]
        self.g = 9.81  # Gravitational acceleration [m/s^2]
        self.Cp = 1004.0  # Specific heat at constant pressure [J/(kg*K)]
        self.z1 = z1  # Measurement Lower altitude (m)
        self.z2 = z2  # Measurement Upper altitude (m)

    ######### Methods ##################################
    def bulk_Richardson_number(self, T1, RH1, T2, RH2, P1, P2, u1, u2):
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
        u1 : float
            Wind speed at lower altitude (m/s)
        u2 : float
            Wind speed at upper altitude (m/s)

        Returns:
        Ri : float
            Bulk Richardson number
        """
        theta1 = self.virtual_potential_temperature(T1, RH1, P1)  # Virtual potential temperature at lower altitude
        theta2 = self.virtual_potential_temperature(T2, RH2, P2)  # Virtual potential temperature at upper altitude
        delta_theta = theta2 - theta1  # Virtual potential temperature difference
        delta_u = u2 - u1  # Wind speed difference
        delta_z = self.z2 - self.z1  # Altitude difference
        mean_theta = (theta1 + theta2) / 2  # Mean virtual potential temperature

        # Calculate the bulk Richardson number
        Ri = (self.g / mean_theta) * (delta_theta / delta_z) / ((delta_u / delta_z) ** 2)
        
        return Ri

    def virtual_potential_temperature(self, T, RH, P):
        """
        Calculate the virtual potential temperature.

        Parameters:
        T : float
            Temperature (K)
        P : float
            Pressure (Pa)

        Returns:
        theta_v : float
            Virtual potential temperature (K)
        """
        # Calculate the mixing ratio
        r = self.mixing_ratio(T, RH, P)  # Assuming 100

        # Calculate the potential temperature
        theta = T * (100000 / P) ** (self.R / self.Cp) * (1 + 0.61 * r)  # Assuming mixing ratio is known or can be calculated
        
        # For simplicity, assuming dry air, so virtual potential temperature equals potential temperature
        theta_v = theta
        
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
        # Calculate the water vapor pressure
        e = self.water_vapor_pressure(T, RH)
        
        # Calculate the mixing ratio
        r = 0.622 * e / (P - e)
        
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
        # Calculate the saturation vapor pressure using the Tetens formula
        T_C = T - 273.15  # Convert temperature to Celsius
        e_s = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * 100  # Saturation vapor pressure in Pa

        # Calculate the actual water vapor pressure
        e = RH / 100 * e_s

        return e
import FileProcessor
import AtmosphericProcessor

src_path = '/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SEALEVEL'
dst_path = '/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SEALEVEL_PROCESSED'
file_name = 'FINO1_2016_2020.nc'
dst_file_name = 'FINO1_2016_2020_processed_v2.pkl'

fp = FileProcessor.FileProcessor(src_path,dst_path)
fp.print_info(file_name)
data_df = fp.read_netcdf(file_name)

z2 = 34
ap = AtmosphericProcessor.AtmosphericProcessor(data_df,z2,z1 = 0)
ap.compute_bulk_Richardson_number()
ap.compute_wind_shear_exponent()

fp.write_pickle(ap.internal_df, dst_file_name)

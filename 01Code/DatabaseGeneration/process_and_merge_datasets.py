import SARProcessor

sar_folder_path = "/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/PROCESSED"
dst_path = "/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/MERGED"
fino1_path = '/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SEALEVEL_PROCESSED/FINO1_2016_2020_processed.pkl'

# SET FINO1 latitude and longitude
lat = 54.0148
lon = 6.5876

# Set target tile dimensions
width = 3500
height = 3500

sp = SARProcessor.SARProcessor(sar_folder_path,dst_path=dst_path,fino_src_path=fino1_path,lat=lat,lon=lon,width=width,height=height)
sp.process_files()
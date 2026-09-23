import FINO1Processor
from SARProcessor import SARProcessor
import Database
import config
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Extract raw primitives from config for parallel computing
sar_src = str(config.sar_src_path)
sar_dst = str(config.sar_dst_path)
db_path = str(config.db_path)
fino_dst_path = str(config.fino_dst_path)
images_path = str(config.images_path)
lat = float(config.lat)
lon = float(config.lon)
width = int(config.width)
height = int(config.height)

# Define worker functions for parallel processing of the DB
# Define worker for SAR processing
def SAR_worker(file_path, sar_src, sar_dst, lat, lon, width, height):
    # Import inside the clean spawned process
    from SARProcessor import SARProcessor

    sp = SARProcessor(
        sar_src_path=sar_src,
        sar_dst_path=sar_dst,
        lat=lat,
        lon=lon,
        width=width,
        height=height,
    )
    sp.process_single_sar_file(file_path)
    return file_path

# Define worker for printing DB images
def DB_worker(idx, db_path, sar_dst, fino_dst_path, images_path):
    # Import inside the clean spawned process
    from Database import Database
    db = Database(db_path,sar_src_path=sar_dst,fino_src_path=fino_dst_path,images_path=images_path)
    db.load_db()
    db.generate_db_images_for_single_product(idx)
    return idx

# Process all FINO1 data files in fino_src_path and store them in fino_dst_path
# fp = FINO1Processor.FINO1Processor(fino_src_path=config.fino_src_path,fino_dst_path=config.fino_dst_path)
# fp.process_fino_files()

# Process all pre-processed SAR SLC files in sar_src_path and store them in sar_dst_path
# sp = SARProcessor.SARProcessor(sar_src_path=config.sar_src_path,sar_dst_path=config.sar_dst_path,lat=config.lat,lon=config.lon,\
#                                 width=config.width,height=config.height)
# sp.process_sar_files()

# 1. Gather all SAR source files
# sar_files = glob.glob(os.path.join(config.sar_src_path, "*.nc"))

# with ProcessPoolExecutor(max_workers=8) as executor:
#     futures = [
#         executor.submit(
#             SAR_worker, f, sar_src, sar_dst, lat, lon, width, height
#         )
#         for f in sar_files
#     ]

#     for future in as_completed(futures):
#         try:
#             print(f"Finished: {future.result()}")
#         except Exception as e:
#             print(f"Error: {e}")

# Merge datasets by assigning the closes measurement to each SAR measurement and generate a database dataframe
db = Database.Database(config.db_path,sar_src_path=config.sar_dst_path,fino_src_path=config.fino_dst_path,images_path=config.images_path)
db.merge_datasets()

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(
            DB_worker, idx, db_path, sar_dst, fino_dst_path, images_path
        )
        for idx, row in db.db.iterrows()
    ]

    for future in as_completed(futures):
        try:
            print(f"Finished: {future.result()}")
        except Exception as e:
            print(f"Error: {e}")
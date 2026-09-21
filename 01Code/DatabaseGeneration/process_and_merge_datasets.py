import FINO1Processor
import SARProcessor
import Database
import config

# Process all FINO1 data files in fino_src_path and store them in fino_dst_path
fp = FINO1Processor.FINO1Processor(fino_src_path=config.fino_src_path,fino_dst_path=config.fino_dst_path)
fp.process_fino_files()

# Process all pre-processed SAR SLC files in sar_src_path and store them in sar_dst_path
sp = SARProcessor.SARProcessor(sar_src_path=config.sar_src_path,sar_dst_path=config.sar_dst_path,lat=config.lat,lon=config.lon,\
                                width=config.width,height=config.height)
sp.process_sar_files()

# Merge datasets by assigning the closes measurement to each SAR measurement and generate a database dataframe
db = Database.Database(config.db_path,sar_src_path=config.sar_dst_path,fino_src_path=config.fino_dst_path)
db.merge_datasets()
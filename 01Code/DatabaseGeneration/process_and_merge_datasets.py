import FINO1Processor
import SARProcessor
import config

fp = FINO1Processor.FINO1Processor(fino_src_path=config.fino_src_path,fino_dst_path=config.fino_dst_path)
fp.process_fino_files()

sp = SARProcessor.SARProcessor(sar_src_path=config.sar_src_path,sar_dst_path=config.sar_dst_path,\
                               fino_src_path=config.fino_dst_path,lat=config.lat,lon=config.lon,\
                                width=config.width,height=config.height)
sp.process_sar_files()
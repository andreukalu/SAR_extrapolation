import FileProcessor
import AtmosphericProcessor

src_path = 'C:/Users/Public/Downloads/FINO1data'
dst_path = 'C:/Users/Public/Downloads'
dst_file_name = 'FINO1_2016_2020_processed_v2.pkl'

fp = FileProcessor.FileProcessor(src_path=src_path,dst_path=dst_path)
fp.read_files()
fp.compute_atmospheric_parameters()
fp.write_pickle(dst_file_name)

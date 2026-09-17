import FileProcessor
import AtmosphericProcessor

src_path = '/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SEALEVEL'
dst_path = '/mnt/csl/datasets/ARS-NEPTUNE/BBDD/SAR/SEALEVEL_PROCESSED'
dst_file_name = 'FINO1_2016_2020_processed.pkl'

fp = FileProcessor.FileProcessor(src_path=src_path,dst_path=dst_path)
fp.read_files()
fp.compute_atmospheric_parameters()
fp.write_pickle(dst_file_name)

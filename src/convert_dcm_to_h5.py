import pandas as pd
from pathlib import Path
from tcia_dcm_to_numpy import convert_dcm_to_png
from tcia_dcm_to_numpy import convert_image_to_numpy


# .dcm to .png
df = pd.read_csv('config.csv', index_col=0, header=0, dtype=str)
dcm_folder = str(df.loc['dcmFolder'].values[0])
png_path = str(df.loc['pathPngs'].values[0])
print('paths from config.csv: ', dcm_folder, png_path)
new_dir_png = Path(png_path)  # makes sure new dir exists
new_dir_png.mkdir(parents=True, exist_ok=True)
convert_dcm_to_png(dcm_folder, png_path)
# .png -> .h5
path_xls_no_polyp = str(df.loc['pathXlsNoPolyp'].values[0])
path_xls_10_polyp = str(df.loc['pathXls10Polyp'].values[0])
path_xls_6_9_polyp = str(df.loc['pathXls69Polyp'].values[0])
new_dir_path = str(df.loc['pathPngs'].values[0])
npy_dir = str(df.loc['pathH5s'].values[0])
print('paths from config.csv: ', path_xls_no_polyp, path_xls_10_polyp, path_xls_6_9_polyp, new_dir_path, npy_dir)
new_dir_npy = Path(npy_dir)  # makes sure new dir exists
new_dir_npy.mkdir(parents=True, exist_ok=True)
convert_image_to_numpy(
    path_xls_no_polyp,
    path_xls_10_polyp,
    path_xls_6_9_polyp,
    new_dir_path,
    npy_dir
)

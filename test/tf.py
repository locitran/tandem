import os
import sys
import pandas as pd 
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import tandem_dimple
from tandem.src.features import TANDEM_FEATS

query = ['P29033 217 Y D', 'P29033 215 I M', 'P29033 214 L V', 'P29033 210 L V', 'P29033 203 I T', 'P29033 197 A T', 'P29033 170 N K', 'P29033 170 N S', 'P29033 168 K R', 'P29033 156 V I', 'P29033 153 V I', 'P29033 127 R H', 'P29033 123 T N', 'P29033 121 I V', 'P29033 115 F V', 'P29033 114 E G', 'P29033 111 I T', 'P29033 107 I L', 'P29033 100 H Q', 'P29033 83 F L', 'P29033 27 V I', 'P29033 16 H Y', 'P29033 4 G V', 'P29033 4 G D', 'P29033 165 R W', 'P29033 34 M T', 'P29033 37 V I', 'P29033 44 W C', 'P29033 44 W S', 'P29033 50 D N', 'P29033 59 G A', 'P29033 75 R Q', 'P29033 75 R W', 'P29033 84 V L', 'P29033 90 L P', 'P29033 95 V M', 'P29033 143 R W', 'P29033 143 R Q', 'P29033 161 F S', 'P29033 163 M T', 'P29033 179 D N', 'P29033 184 R Q', 'P29033 195 M T', 'P29033 197 A S', 'P29033 202 C F', 'P29033 205 L V', 'P29033 206 N S']
labels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
job_name = 'tf_GJB2'
seed = 73

# df_gjb2 = pd.read_csv('/mnt/nas_1/YangLab/loci/tandem/data/GJB2/final_features.csv')
# fm = df_gjb2[df_gjb2['SAV_coords'].isin(query)][TANDEM_FEATS['v1.1']]
# fm = fm.to_records(index=False)

t = tandem_dimple(
    query,
    job_name=job_name,
    features=None, 
    tf_name=job_name,
    labels=labels,
    config={'seed':seed}, 
    custom_PDB='2ZW3', 
    featSet=None,
    refresh=False,
    pkl_folder='data',
)
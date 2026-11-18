import pandas as pd
import numpy as np
import os
from pathlib import Path
from mtx.utils.config import load_config
from mtx.data.extract_transformer_from_db import get_transformer_data
from mtx.data.transformers_with_structures import merge_transformer_and_structure
#from mtx.data.extract_ami_data import ami_data_from_db

### --- Loaders --- ###

## Pickle Path Directory
base_dir = Path.cwd()
dir = base_dir / "pickles"
    
def load_transformers_data():
    
    path = dir / "Transformers_With_Structures.pkl"
    if os.path.exists(path):
        df = pd.read_pickle(path)
        return clean_transformer_data(df)
    else:
        df = merge_transformer_and_structure()
        df.to_pickle(path)
        return clean_transformer_data(df)

def load_ami_data():
    path = dir / "NY_CMP_AMI_Data.pkl"
    if os.path.exists(path):
        df = pd.read_pickle(path)
        return df
    else:
        df = ami_data_from_db()
        df.to_pickle(path)
        return df

def clean_transformer_data(df):
    df = df.reset_index(drop=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace({"": 'NULL', np.nan: 'NULL'}, inplace=True)
    df.replace("NULL", np.nan, inplace=True)  # Restore NaN for calculations
    return df


def clean_meter_data(df):
    return df.drop_duplicates(subset=['MeterID']).reset_index(drop=True)


def aggregate_sap_ids(df):
    df['SAPID'] = df['SAPID'].astype(str)
    grouped = df.groupby('MeterID', as_index=False).agg({'SAPID': ', '.join})
    grouped['multiple'] = grouped['SAPID'].str.contains(',')
    grouped['SAPID'] = grouped['SAPID'].replace('nan', 'No Xfmr FLOC match')
    return grouped


### --- Matching Function --- ###

def match_meters_to_transformers():
    path = dir / 'NY_CMP_Match.pkl'
    if os.path.exists(path):
        df = pd.read_pickle(path)
        return df
    else:
        df_meters = load_ami_data()
        df_transformers = load_transformers_data()
        merged = df_meters.merge(
            df_transformers,
            how='left',
            left_on='SuperiorFLOC',
            right_on='FuncLoc'
        )

        sapid_agg = aggregate_sap_ids(merged)

        merged.drop(columns=['SAPID'], inplace=True, errors='ignore')
        merged = merged.merge(sapid_agg[['MeterID', 'SAPID']], how='left', on='MeterID')

        merged.rename(columns={
            'SuperiorFLOC': 'MeterFLOC',
            'FuncLoc': 'XfmrFLOC',
            'LONGITUDE': 'MeterX',
            'LATITUDE': 'MeterY',
            'POINT_X': 'XfmrX',
            'POINT_Y': 'XfmrY',
            'OPCO_x':'OPCO'
        }, inplace=True)

        final_cols = [
            'MeterID', 'Circuit', 'SAPID', 'MeterFLOC', 'XfmrFLOC',
            'SAP_Address', 'SAP_Town', 'ZIP', 'DivisionName', 'OPCO',
            'OperatingVoltage', 'POLEID', 'TransformerType',
            'MeterX', 'MeterY', 'XfmrX', 'XfmrY'
        ]

        df_matched = merged[final_cols].drop_duplicates(subset=['MeterID']).reset_index(drop=True)
        condition1 = ~df_matched['XfmrFLOC'].isnull()
        condition2 = df_matched['SAPID'] == 'No Xfmr FLOC match'
        combined_condition = condition1 & condition2
        # Set 'SAPID' to null where the condition is met
        df_matched.loc[combined_condition, 'SAPID'] = 'NULL'

        df_matched['SAPID'] = df_matched['SAPID'].replace(r'\bnan\b', 'NULL', regex=True)
        df_matched.to_pickle(path)
        return df_matched

if __name__ == '__main__':
    df_matched = match_meters_to_transformers()

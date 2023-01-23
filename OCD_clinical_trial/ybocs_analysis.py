# This is just a script to save the XLS master file (with behavioral data) into a pandas dataframe.

import argparse
import bct
import h5py
import importlib
import itertools
import joblib
from joblib import Parallel, delayed
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sbn
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.decomposition import PCA
import statsmodels
from statsmodels.stats import multitest
import sys
import time
from time import time

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_clinical_trial'
baseline_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

sys.path.insert(0, os.path.join(baseline_dir))
sys.path.insert(0, os.path.join(baseline_dir, 'old'))
sys.path.insert(0, os.path.join(baseline_dir, 'utils'))

sys.path.insert(0, os.path.join(proj_dir, 'code', 'OCD_clinical_trial', 'functional'))
from seed_to_voxel_analysis import get_group


atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(proj_dir, 'code/patients_list.txt'), names=['name'])['name']


xls_fname = 'P2253_Data_Master-File.xlsx' #'P2253_OCD_Data_Pre-Post-Only.xlsx' #'P2253_YBOCS.xlsx'


def create_dataframes(args):
    """ load XLS master file and currate into controls and patients pandas dataframes  """
    xls = pd.read_excel(os.path.join(proj_dir, 'data', xls_fname), sheet_name=['OCD Patients'])

    df_pat = xls['OCD Patients'][['Participant_ID', 'Pre/Post/6mnth', 'Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total', 'Anx_total', 'Dep_Total', 'FSIQ-4_Comp_Score', 'Medications']]
    #df_pat = df_pat[df_pat['Pre/Post/6mnth']=='Pre'][['Participant_ID', 'Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total', 'Anx_total', 'Dep_Total', 'FSIQ-4_Comp_Score', 'Medications']]

    # sort alphabetically by subject ID
    df_pat['subj'] = ['sub-patient{:2s}'.format(s.split('_')[0][-2:]) for s in df_pat.Participant_ID]
    df_pat.sort_values(by=['subj'], inplace=True)
    df_pat['group'] = [get_group(subj) for subj in df_pat.subj]

    df_pat.rename(columns={'Pre/Post/6mnth': 'ses'}, inplace=True)
    df_pat.replace({'ses': {'Pre':'ses-pre', 'Post':'ses-post'}}, inplace=True)
    df_pat.drop(df_pat[df_pat['ses'] == '6mnth'].index, inplace=True)
    df_pat.reset_index(drop=True, inplace=True)

    # savings
    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'df_pat.pkl'), 'wb') as f:
            pickle.dump(df_pat, f)

    return df_pat




def print_medications(df_pat, args=None):
    """ print medications taken by patients """
    df_med = df_pat[(df_pat['Medications']!=9999)]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--print_medications', default=False, action='store_true', help='print subjects medications')
    args = parser.parse_args()

    #revoked=['sub-patient14', 'sub-patient15', 'sub-patient16', 'sub-patient29', 'sub-patient35', 'sub-patient51']
    revoked = []
    df_pat = create_dataframes(args)

    if args.print_medications:
        print_medications(df_pat, args)

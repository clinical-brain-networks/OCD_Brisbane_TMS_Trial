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
import pickle
import pingouin as pg
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
code_dir = os.path.join(proj_dir, 'code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(proj_dir, 'code/patients_list.txt'), names=['name'])['name']


xls_fname = 'P2253_Data_Master-File.xlsx' #'P2253_OCD_Data_Pre-Post-Only.xlsx' #'P2253_YBOCS.xlsx'

groups = ['group1', 'group2']
df_groups = pd.read_csv(os.path.join(proj_dir, 'data', 'groups.txt'), \
                        sep=' ', index_col=False, dtype=str, encoding='utf-8')
group_colors = {'group1': 'orange', 'group2':'lightslategray'}

# checklist dimensions
checklist_13dims = ['Aggressive', 'Contamination', 'Sexual', 'Hoarding/Saving', 'Religious', 'Symmetry', 'Miscellaneous Obs', 
                        'Somatic', 'Cleaning', 'Checking', 'Repeating', 'Counting', 'Ordering', 'Hoarding/collecting', 'Miscellaneous Comp']

checklist_5dims = ['Sexual/Religious', 'Symmetry/Ordering', 'Hoarding', 'Contamination/Cleaning', 'Aggressive/Checking']

checklist_2dims = ['Obsessions', 'Compulsions']

def get_group(subj):
    group = df_groups[df_groups.subj==subj].group
    if len(group):
        return group.values[0]
    else:
        return np.NaN


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



def get_obsession_compulsion_scores(df, dims, option='sum'):
    """ Extracts compulsions and obsessions dimensions from YBOCS 
        inputs:
            df: pandas dataframe of YBOCS scores 
            options: defines how to compute scores based on raw values ('sum' or 'mean')
    """
    for dim in dims:
        k = np.zeros((len(df),))
        n_k = 0
        for col in df.columns.to_list():
            if (dim.lower() in col.lower()):
                k += np.array(df[col], dtype=int)
                n_k += 1
        if option=='sum':
            df[dim] = k
        elif option=='mean':
            df[dim] = k/n_k
        else:
            NameError("Option must be 'mean' or 'sum' to extract obsession/compulsion scores from YBOCS checklist")
    return df


def get_5dims_scores(df, checklist_5dims, checklist_13dims, option='sum'):
    """ Extracts 5 dimensions from YBOCS checklists
        inputs:
            df: pandas dataframe of YBOCS scores 
            cheklist_5dims: list of 5 dimensions
            cheklist_13dims: list of 13 intermediate dimensions
            options: defines how to compute scores based on raw values ('sum' or 'mean')
    """
    for k13 in checklist_13dims:
        for k in df.columns.to_list():
            if k13 in k:
                df.rename(columns={k:k13}, inplace=True)
    df.dropna(inplace=True)

    for k5 in checklist_5dims:
        dim = np.zeros((len(df),))
        n_dims = 0
        for k13 in checklist_13dims:
            if ((k13 in k5) or (k5 in k13)):
                dim += np.array(df[k13], dtype=int)
                n_dims += 1
        if option=='sum':
            df[k5] = dim
        elif option=='mean':
            df[k5] = dim/n_dims
        else:
            print("option must be 'mean' or 'sum' ")
    return df

def fix_session_entries(df):
    """ fix some entries which have typo/spaces """
    valid_ses = ['Pre', 'Post', '6mnth']
    real_ses = {'Pre':'ses-pre', 'Post':'ses-post', '6mnth':'6mnth'}

    def get_ses(session):
        for ses in valid_ses:
            if ses==session:
                return ses
            else:
                return np.NaN
        
    df['session'] = [get_ses(session) for session in df['Pre/Post/6mnth']]
    return df.dropna()


def create_df_ybocs_dims():
    """ extract columns with YBOCS dimensions """
    xls = pd.read_excel(os.path.join(proj_dir, 'data', 'P2253_Data_Master-File.xlsx'), sheet_name=['OCD Patients', 'Healthy Controls'])

    checklist_cols = ['YBOCS SC Aggressive Obsessions', 'YBOCS SC Contamination Obsessions', 'YBOCS SC Sexual Obsessions', 
                    'YBOCS SC Hoarding/Saving Obsessions', 'YBOCS SC Religious Obsessions', 'YBOCS SC Symmetry/Exactness Obsessions', \
                    'YBOCS SC Miscellaneous Obsessions', 'YBOCS SC Somatic Obsessions', 'YBOCS SC Cleaning/Washing Compulsions', 
                    'YBOCS SC Checking Compulsions', 'YBOCS SC Repeating Compulsions', 'YBOCS SC Counting Compulsions', 
                    'YBOCS SC Ordering/Arranging Compulsions', "YBOC's SC Hoarding/collecting compulsions", 'YBOCS SC Miscellaneous Compulsions']


    other_cols = ['Participant_ID', 'Pre/Post/6mnth', 'Age', 'Gender(F=1,M=2)', 'Handedness(R=1,L=2)', 'YBOCS_Total', 'OBQ_Total', 'HAMA_Total', 'MADRS_Total', 'OCIR_Total']

    df_pat = xls['OCD Patients'][np.concatenate([other_cols, checklist_cols])]    
    df_pat.sort_values(by=['Participant_ID'], inplace=True)
    df_pat['subj'] = ['sub-patient{:2s}'.format(s.split('_')[0][-2:]) for s in df_pat.Participant_ID]
    df_pat['group'] = [get_group(subj) for subj in df_pat['subj']]
    df_pat = df_pat.dropna()
    df_pat = get_obsession_compulsion_scores(df_pat, checklist_2dims, option='sum')
    df_pat = get_5dims_scores(df_pat, checklist_5dims, checklist_13dims, option='sum')
    df_pat = fix_session_entries(df_pat)
    return df_pat.dropna()

def print_stat_pre_post_between_groups(df, dims):
    df_pre = df[df['Pre/Post/6mnth']=='Pre']
    df_post = df[df['Pre/Post/6mnth']=='Post']

    for dim in dims:
        diff_g1 = np.array(df_pre[df_pre.group=='group1'][dim]) - np.array(df_post[df_post.group=='group1'][dim])
        diff_g2 = np.array(df_pre[df_pre.group=='group2'][dim]) - np.array(df_post[df_post.group=='group2'][dim])
        t,p = scipy.stats.ttest_ind(diff_g1, diff_g2)
        print("Pre-post difference in {} between active and sham: t={:.2f}, p={:.2f}".format(dim,t,p))

        print("\n"+dim+':')
        mixed = pg.mixed_anova(data=df.dropna(), dv=dim, within='session', between='group', subject='subj')
        pg.print_table(mixed)
        posthocs = pg.pairwise_ttests(data=df.dropna(), dv=dim, within='session', between='group', subject='subj')
        pg.print_table(posthocs)



def plot_ybocs_dims_to_fc(df):
    """ simple scatter plot of FC to YBOCS dimensions relation with printing of statistics (correlation) """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_voi_corr_detrend_gsr_filtered_scrubFD05_Harrison2009_brainFWHM8mm_bilateral_indStimSite_10mm_diameter.pkl'), 'rb') as f: 
        df_voi_corr = pickle.load(f)

    df_ybocs = df[(df.session=='Pre')][np.concatenate([['subj'],checklist_5dims])]
    df_diff_fc = df_voi_corr[df_voi_corr.ses=='pre-post'][['subj', 'group', 'corr']]

    df_stats = df_ybocs.merge(df_diff_fc).dropna()

    plt.figure(figsize=[15,3])
    for i,dim in enumerate(checklist_5dims):
        plt.subplot(1,5,i+1)
        for group in groups:
            X = df_stats[df_stats.group==group]['corr']
            Y = df_stats[df_stats.group==group][dim]
            r,p = scipy.stats.pearsonr(X, Y)
            print("{} {} r={:.2f}, p={:.2f}".format(dim, group, r, p))
            plt.scatter(X, Y, color=group_colors[group])
        plt.title(dim)
        plt.xlabel('FC')
        plt.ylabel('score')
        plt.tight_layout()


def print_ybocs_dims_table(df):
    """ severity (absent-mild-severe) of YBOCS dimensions per group """
    alpha = 0.5
    plt.figure(figsize=[12,4])
    for i,group in enumerate(groups):
        df_grp = df[df.group==group]
        n = len(df_grp)
        plt.subplot(1,2,i+1)
        for i,k5 in enumerate(checklist_5dims):
            absent = np.sum(df_grp[k5]<1)/n
            mild = np.sum((df_grp[k5]>=1) & (df_grp[k5]<2))/n
            severe = np.sum(df_grp[k5]>=2)/n
            plt.barh(i, absent, color='blue', alpha=alpha)
            plt.barh(i, mild, left=absent, color='orange', alpha=alpha)
            plt.barh(i, severe, left=absent+mild, color='red', alpha=alpha)
            print("{} {} \t {:.3f}  {:.3f}  {:.3f}".format(group, k5, absent, mild, severe))
        plt.legend(['absent', 'mild', 'severe'], loc='upper right')
        plt.xticks(np.arange(0,1.1,0.2), labels=[str(int(k*100)) for k in np.arange(0,1.1,0.2)])
        plt.xlabel('%')
        plt.yticks(np.arange(i+1), labels=checklist_5dims)
        plt.title(group) 
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--print_medications', default=False, action='store_true', help='print subjects medications')
    parser.add_argument('--print_ybocs_stats', default=False, action='store_true', help='print stats related to YBOCS dimensions')
    parser.add_argument('--plot_ybocs_dims_to_fc', default=False, action='store_true', help='print stats and scatter plot of correlation between YBOCS dimensions and FC')
    parser.add_argument('--print_ybocs_dims_table', default=False, action='store_true', help='print YBOCS dimension table and stacked bar chart')
    args = parser.parse_args()

    #revoked=['sub-patient14', 'sub-patient15', 'sub-patient16', 'sub-patient29', 'sub-patient35', 'sub-patient51']
    revoked = []
    df_pat = create_dataframes(args)

    if args.print_medications:
        print_medications(df_pat, args)

    df = create_df_ybocs_dims()
    if args.print_ybocs_stats:
        print_stat_pre_post_between_groups(df, checklist_2dims)
    if args.plot_ybocs_dims_to_fc:
        plot_ybocs_dims_to_fc(df)
    if args.print_ybocs_dims_table:
        print_ybocs_dims_table(df[df.session=='Pre'].dropna())


# Script to perform FC analysis based on seed or parcellation to voxel correlations
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
import bct
from datetime import datetime
import glob
import gzip
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
from nilearn import datasets
from nilearn.image import load_img, new_img_like, resample_to_img, binarize_img, iter_img, math_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_img, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
import nltools
import numpy as np
import os
import pickle
import pandas as pd
import pdb
import scipy
from scipy.io import loadmat
from scipy import ndimage
import seaborn as sbn
import shutil
import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import statsmodels
from statsmodels.stats import multitest
import sys
import time
from time import time
import pdb
import platform
import warnings
warnings.filterwarnings('once')

# special imports
from OCD_baseline.old import qsiprep_analysis
from OCD_baseline.utils import atlaser
from OCD_baseline.structural.voxelwise_diffusion_analysis import cohen_d


# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
    import pingouin as pg
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

# general paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCD_clinical_trial'
deriv_dir = os.path.join(proj_dir, 'data/derivatives')

baseline_dir = working_dir+'lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(baseline_dir, 'docs/code')
atlas_dir = os.path.join(baseline_dir, 'utils')


atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)

# Harrison 2009 seed locations:
seed_loc = {'AccR':[9,9,-8], 'AccL':[-9,9,-8] }
        #'dPutL':[-28,1,3], 'dPutR':[28,1,3], \
        #'vPutL':[-20,12,-3] , 'vPutR':[20,12,-3]} #, \
        #'dCaudL':[-13,15,9], 'dCaudR':[13,15,9]} #, \
        #'vCaudSupL':[-10,15,0], 'vCaudSupR':[10,15,0], \
        #'drPutL':[-25,8,6], 'drPutR':[25,8,6]}


groups = ['group1', 'group2']

pathway_mask = {'Acc':['OFC', 'PFClv', 'PFCv'],
                'dCaud':['PFCd_', 'PFCmp', 'PFCld_'],
                'dPut':['Ins', 'SomMotB_S2'], #'PFCld_''PFCl_',
                'vPut':['PFCl_', 'PFCm'],
                'NucleusAccumbens':['OFC', 'PFClv', 'PFCv'],} #'PFCd'

cut_coords = {'Acc':[25,57,-6],
              'dCaud':None,
              'dPut':[50,11,19],
              'vPut':[-25,56,35],
              'NucleusAccumbens':[25,57,-6]}

df_groups = pd.read_csv(os.path.join(proj_dir, 'data', 'groups.txt'), \
                        sep=' ', index_col=False, dtype=str, encoding='utf-8')

stim_radius = 5 # radius of sphere around stim site
stim_coords_xls_fname = 'MNI_coordinates_FINAL.xlsx'
stim_coords = pd.read_excel(os.path.join(proj_dir, 'data', stim_coords_xls_fname), usecols=['P ID', 'x', 'y', 'z'])
stim_coords['subjs'] = stim_coords['P ID'].apply(lambda x : 'sub-patient'+x[-2:])

seed_suffix = { 'Harrison2009': 'sphere_seed_to_voxel',
                'TianS4':'seed_to_voxel'}
seed_ext =  { 'Harrison2009': '.nii.gz',
                'TianS4':'.nii.gz'}

group_colors = {'group1': 'orange', 'group2':'lightslategray'}

pointplot_ylim = {  'Harrison2009': {'corr': [-1,1], 'fALFF':[1,2]},
                    'TianS4': {'corr': [-0.5,0.5], 'fALFF':[1,2]},
                }


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def get_seed_names(args):
    if args.seed_type == 'Harrison2009':
        seeds = list(seed_loc.keys()) #['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']
        subrois = np.unique([seed[:-1] for seed in seeds])#['Acc', 'dCaud', 'dPut', 'vPut', 'drPut']
    else:
        seeds = ['NucleusAccumbens']
        subrois = ['NucleusAccumbens']
    return seeds, subrois


def get_group(subj):
    group = df_groups[df_groups.subj==subj].group
    if len(group):
        return group.values[0]
    else:
        return 'none'


def seed_to_voxel(subj, ses, seeds, metrics, atlases, args=None):
    """ perform seed-to-voxel analysis of bold data based on atlas parcellation """
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for metric in metrics:
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        bold_file = os.path.join(deriv_dir, 'post-fmriprep-fix', subj, ses, 'func', \
                                 subj+'_'+ses+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz')
        if os.path.exists(bold_file):
            bold_img = nib.load(bold_file)
        else:
            print("{} {} bold file not found, skip".format(subj, ses))
            break
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, verbose=0, standardize='zscore')
        voxels_ts = brain_masker.fit_transform(bold_img)

        for atlas in atlases:
            # get atlas utility
            atlazer = atlaser.Atlaser(atlas)

            # extract seed timeseries and perform seed-to-voxel correlation
            for seed in seeds:
                seed_img = atlazer.create_subatlas_img(seed)
                seed_masker = NiftiLabelsMasker(seed_img, t_r=0.81, \
                    low_pass=0.1, high_pass=0.01 ,standardize='zscore')
                seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
                seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/voxels_ts.shape[0]
                seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr.mean(axis=-1).T)
                fname = '_'.join([subj,ses,metric,args.fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz'])
                nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))

    print('{} seed_to_voxel performed in {}s'.format(subj,int(time()-t0)))


# TODO: could refactor this function, only a few lines changed from the one above
def sphere_seed_to_voxel(subj, ses, seeds, metrics, atlases=['Harrison2009'], args=None):
    """ perform seed-to-voxel analysis of bold data using Harrison2009 3.5mm sphere seeds """
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for atlas,metric in itertools.product(atlases,metrics):
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        bold_file = os.path.join(deriv_dir, 'post-fmriprep-fix', subj, ses, 'func', \
                                 subj+'_'+ses+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz')
        if os.path.exists(bold_file):
            bold_img = nib.load(bold_file)
        else:
            print("{} {} bold file not found, skip".format(subj, ses))
            break
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, verbose=0, standardize='zscore')
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for seed in seeds:
            seed_masker = NiftiSpheresMasker([np.array(seed_loc[seed])], radius=3.5, t_r=0.81, \
                                low_pass=0.1, high_pass=0.01, verbose=0, standardize='zscore')
            seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
            seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/voxels_ts.shape[0]
            seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr)
            fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
            fname = '_'.join([subj,ses,metric,fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz'])
            nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} seed_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))


def merge_LR_hemis(subjs, seeds, seses, metrics, seed_type='sphere_seed_to_voxel', args=None):
    """ merge the left and right correlation images for each seed in each subject """
    if args.seed_type=='Harrison2009':
        hemis = ['L', 'R']
    else:
        hemis = ['']
    in_fnames = dict( ( ((seed,metric),[]) for seed,metric in itertools.product(seeds,metrics) ) )
    for atlas,metric,ses in itertools.product(args.atlases, metrics,seses):
        for i,seed in enumerate(seeds):
            for k,subj in enumerate(subjs):
                group = get_group(subj)
                if group=='none':
                    subjs.drop(subjs[subjs==subj].index[0], inplace = True)
                    print(subj+" removed because does not belong to any group")
                    continue
                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                fnames = [os.path.join(proj_dir, 'postprocessing', subj,
                                       '_'.join([subj,ses,metric,fwhm,atlas,seed+hemi,seed_suffix[args.seed_type],'corr.nii.gz']))
                          for hemi in hemis]
                if os.path.exists(fnames[0]):
                    new_img = nilearn.image.mean_img(fnames)
                else:
                    print("{} not found, skip".format(fnames[0]))
                    break
                #fname = s+'_detrend_gsr_filtered_'+seed+'_sphere_seed_to_voxel_corr.nii'
                fname = '_'.join([subj,ses,metric,fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz'])
                os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, group), exist_ok=True)
                nib.save(new_img, os.path.join(args.in_dir, metric, fwhm, seed, group, fname))
                in_fnames[(seed,metric)].append(os.path.join(args.in_dir, metric, fwhm, seed, group, fname))
    print('Merged L-R hemishperes')
    return in_fnames


def unzip_correlation_maps(subjs, seses, metrics, atlases, seeds, args):
    """ extract .nii files from .nii.gz and put them in place for analysis with SPM (not used if only analysing with nilearn) """
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    [os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, grp), exist_ok=True) \
            for metric,seed,grp in itertools.product(metrics,seeds,groups)]

    print('Unzipping seed-based correlation maps for use in SPM...')

    for subj,ses,metric,atlas,seed in itertools.product(subjs,seses,metrics,atlases,seeds):
        fname = '_'.join([subj,ses,metric,fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz'])
        infile = os.path.join(proj_dir, 'postprocessing', subj, fname)
        group = get_group(subj)
        with gzip.open(infile, 'rb') as f_in:
            with open(os.path.join(args.in_dir, ses, metric, fwhm, seed, group, fname[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_subjs_after_scrubbing(subjs, seses, metrics, min_time=5):
    scrub_key = 'scrubbed_length_min'
    scrub_thr = min_time
    proc_dir = 'post-fmriprep-fix'
    d_dir = deriv_dir 

    revoked = []
    for subj,ses,metric in itertools.product(subjs,seses,metrics):
        fname = 'fmripop_'+metric+'_parameters.json'
        fpath = os.path.join(d_dir, proc_dir, subj, ses, 'func', fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                f_proc = json.load(f)
                if f_proc['scrubbing']:
                    if f_proc[scrub_key] < scrub_thr:
                        print("{} has less than {:.2f} min of data left after scrubbing, removing it..".format(subj, f_proc[scrub_key]))
                        revoked.append(subj)
        else:
            print("{} preprocessing not found, removing it..".format(subj))
            revoked.append(subj)

    rev_inds = [np.where(s==subjs)[0][0] for s in revoked]
    subjs = subjs.drop(np.unique(rev_inds))
    return subjs, np.unique(revoked)



def resample_masks(masks):
    """ resample all given masks to the affine of the first in list """
    ref_mask = masks[0]
    out_masks = [ref_mask]
    for mask in masks[1:]:
        out_masks.append(resample_to_img(mask, ref_mask, interpolation='nearest'))
    return out_masks

def mask_imgs(flist, masks=[], seed=None, args=None):
    """ mask input images using intersection of template masks and pre-computed within-groups union mask """
    # mask images to improve SNR
    t_mask = time()
    if args.use_gm_mask:
        gm_mask = datasets.load_mni152_gm_mask()
        masks.append(binarize_img(gm_mask))
    if args.use_fspt_mask: ## not sure it works fine
        fspt_mask = load_img(os.path.join(baseline_dir, 'utils', 'Larger_FrStrPalThal_schaefer400_tianS4MNI_lps_mni.nii'), dtype=np.float64)
        masks.append(binarize_img(fspt_mask))
    if args.use_cortical_mask:
        ctx_mask = load_img(os.path.join(baseline_dir, 'utils', 'schaefer_cortical.nii'), dtype=np.float64)
        masks.append(binarize_img(ctx_mask))
    if args.use_frontal_mask:
        Fr_node_ids, _ = qsiprep_analysis.get_fspt_Fr_node_ids('schaefer400_tianS4')
        atlazer = atlaser.Atlaser(atlas='schaefer400_tianS4')
        Fr_img = atlazer.create_brain_map(Fr_node_ids, np.ones([len(Fr_node_ids),1]))
        masks.append(binarize_img(Fr_img))
    if args.use_seed_specific_mask:
        atlazer = atlaser.Atlaser(atlas='schaefer400_tianS4')
        frontal_atlas = atlazer.create_subatlas_img(rois=pathway_mask[seed])
        masks.append(binarize_img(frontal_atlas))
    if masks != []:
        masks = resample_masks(masks)
        mask = nilearn.masking.intersect_masks(masks, threshold=1, connected=False) # thr=1 : intersection; thr=0 : union
        masker = NiftiMasker(mask)
        masker.fit(imgs=list(flist))
        masker.generate_report() # use for debug
        masked_data = masker.transform(imgs=flist.tolist())
        imgs = masker.inverse_transform(masked_data)
        imgs = list(iter_img(imgs))  # 4D to list of 3D
    else:
        imgs = list(flist)
        masker=None
        mask = None
    print('Masking took {:.2f}s'.format(time()-t_mask))
    return imgs, masker, mask


def threshold_contrast(contrast, height_control='fpr', alpha=0.005, cluster_threshold=10):
    """ cluster threshold contrast at alpha with height_control method for multiple comparisons """
    thresholded_img, thresh = threshold_stats_img(
        contrast, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
    cluster_table = get_clusters_table(
        contrast, stat_threshold=thresh, cluster_threshold=cluster_threshold,
        two_sided=True, min_distance=5.0)
    return thresholded_img, thresh, cluster_table



def get_subj_stim_mask(subj, args):
    """ create sphere mask around stim stim for individuals """
    l = stim_coords[stim_coords['subjs']==subj]
    if l.empty:
        print(subj+' not in file '+stim_coords_xls_fname)
        return None,None
    stim_mask = nltools.create_sphere(np.array([l['x'], l['y'], l['z']]).flatten(), radius=args.stim_radius)
    stim_masker = NiftiSpheresMasker([np.array([l['x'], l['y'], l['z']]).flatten()], radius=args.stim_radius,
                                     smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, low_pass=0.25, standardize=False)
    return stim_mask, stim_masker


def compute_voi_corr(subjs, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ compute correlation between seed and VOI for each pathway, to extract p-values, effect size, etc. """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            group = get_group(subj)
            if group == 'none':
                print('{} not in group list, removed it.'.format(subj))
                continue;
            # get VOI mask
            if args.use_group_avg_stim_site:
                voi_mask = load_img(os.path.join(proj_dir, 'utils', 'mask_stim_VOI_5mm.nii.gz'))
            else:
                voi_mask,_ = get_subj_stim_mask(subj, args)
            if voi_mask == None:
                continue
            # compute correlation
            for seed in seeds:
                pre = post = 0
                for ses in args.seses:
                    # load correlation map
                    if args.unilateral_seed:
                        fname = '_'.join([subj, ses, metric, fwhm, atlas, seed, seed_suffix[args.seed_type], 'corr'+seed_ext[args.seed_type]])
                        fpath = os.path.join(proj_dir, 'postprocessing', subj, fname)
                    else:
                        fname = '_'.join([subj, ses, metric, fwhm, atlas, seed, seed_suffix[args.seed_type], 'corr'+seed_ext[args.seed_type]])
                        fpath = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs', args.seed_type, 'seed_not_smoothed',
                                        metric, fwhm, seed, group, fname)
                    if os.path.exists(fpath):
                        corr_map = load_img(fpath)
                    else:
                        print("{} {} FC file not found, skip.\n{}".format(subj, ses, fpath))
                        pre=np.nan
                        post=np.nan
                        if ses=='ses-post':
                            dfs = dfs[:-1]
                        break

                    # load voi mask
                    #voi_mask = load_img(os.path.join(proj_dir, 'utils', 'frontal_'+seed+'_mapping_AND_mask_stim_VOI_5mm.nii.gz'))
                    voi_mask = resample_to_img(voi_mask, corr_map, interpolation='nearest')
                    # extract correlations
                    voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                    avg_corr = np.mean(voi_corr[voi_corr!=0])
                    df_line = {'subj':subj, 'ses':ses, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'group':group, 'pathway':'_'.join([seed,'to','stim']), 'corr':avg_corr}
                    dfs.append(df_line)

                    # simple way to add pre - post difference to compute interaction
                    #----------------------- < from here
                    if ses=='ses-pre':
                        pre = avg_corr
                    else:
                        post = avg_corr
                df_line = {'subj':subj, 'ses':'pre-post', 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'group':group, 'pathway':'_'.join([seed,'to','stim']), 'corr':pre-post}
                dfs.append(df_line)
                #--------------------------- < to here
    df_voi_corr = pd.DataFrame(dfs)
    return df_voi_corr



def plot_voi_corr(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ violinplots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.5, 0.5]
    fig = plt.figure(figsize=[18,6])
    
    # 1 row per seed, 4 columns: group, pre-post, group1 pre-post, group2 pre-post
    for i,seed in enumerate(seeds):
      # group difference
      ax = plt.subplot(i+1,3,3*i+1)
      tmp_df = df_voi_corr[(df_voi_corr['pathway']=='_'.join([seed,'to','stim'])) & (df_voi_corr['ses']!='pre-post')]
      sbn.barplot(data=tmp_df, y='corr', x='pathway', hue='group', orient='v')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      #ax.get_legend().set_visible(False)
      plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      ax.set_title(seed+' - group difference')

      # pre-post difference across groups
      ax = plt.subplot(i+1,3,3*i+2)
      tmp_df = df_voi_corr[(df_voi_corr['pathway']=='_'.join([seed,'to','stim'])) & (df_voi_corr['ses']=='pre-post')]
      sbn.barplot(data=tmp_df, y='corr', x='pathway', hue='group', orient='v')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      #ax.get_legend().set_visible(False)
      plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      ax.set_title(seed+' - pre-post difference')

      # pre-post and groups
      ax = plt.subplot(i+1,3,3*i+3)
      tmp_df = df_voi_corr[(df_voi_corr['pathway']=='_'.join([seed,'to','stim'])) & (df_voi_corr['ses']!='pre-post')]
      sbn.barplot(data=tmp_df, y='corr', x='group', hue='ses', orient='v')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

      plt.tight_layout()

    if args.save_figs:
        if args.use_group_avg_stim_site:
            suffix = '_radius5mm_avg'
        else:
            suffix = '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2))
        figname = 'seed_to_stim_VOI'+suffix+'_group_by_session.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))
        #plt.savefig(os.path.join('/home/sebastin/tmp/', figname))


def print_voi_stats(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ print seed to VOI stats """
    print('Seed to VOI statistics:\n-------------------------')
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
        out = dict()
        for seed in seeds:
            key = '_'.join([seed, 'to', 'stim'])
            df_con = df_voi_corr.loc[ (df_voi_corr['group']=='group1')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key)
                                    & (df_voi_corr['ses']=='pre-post') ]
            df_pat = df_voi_corr.loc[ (df_voi_corr['group']=='group2')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key)
                                    & (df_voi_corr['ses']=='pre-post') ]
            t,p = scipy.stats.ttest_ind(df_con['corr'], df_pat['corr'])
            d = cohen_d(df_con['corr'], df_pat['corr'])
            print("{} {} {} {} pre-post  T={:.3f}   p={:.3f}   cohen's d={:.2f}".format(atlas,metric,fwhm,key,t,p,d))




def get_file_lists(subjs, seed, atlas, metric, args):
    """ returns 3 file lists corresponding to controls, patients, and combined
    controls+patients paths of imgs to process """
    # naming convention in file system
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    # get images path
    group1_flist = []
    group2_flist = []
    if args.group_by_session:
        for ses in args.seses:
            cl = np.sort(glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'group1', '*'+ses+'*')))
            group1_flist.append(cl)
            pl = np.sort(glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'group2', '*'+ses+'*')))
            group2_flist.append(pl)

    # compute pre-post, simplifies the design matrix
    else:
        for subj in subjs:
            if subj not in args.revoked:
                grp = get_group(subj)
                pre_img = os.path.join(args.in_dir, metric, fwhm, seed, grp,
                                        '_'.join([subj,'ses-pre',metric,fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz']))
                post_img = os.path.join(args.in_dir, metric, fwhm, seed, grp,
                                        '_'.join([subj,'ses-post',metric,fwhm,atlas,seed,seed_suffix[args.seed_type],'corr.nii.gz']))
                sub_img = math_img('img1 - img2', img1=pre_img, img2=post_img)
                if grp=='group1':
                    group1_flist.append(sub_img)
                elif grp=='group2':
                    group2_flist.append(sub_img)
                else:
                    continue;
    group1_flist = np.array(group1_flist).flatten()
    group2_flist = np.array(group2_flist).flatten()
    # remove revoked subjects -- do controls and patients separately on purpose
    if list(args.revoked) != []:
        group1_flist = [l for l in group1_flist if ~np.any([s in l for s in args.revoked])]
        group2_flist = [l for l in group2_flist if ~np.any([s in l for s in args.revoked])]
    flist = np.hstack([group1_flist, group2_flist]).flatten()
    return group1_flist, group2_flist, flist





def compute_ALFF(subj, args=None):
    """ compute Amplitude Low Frequency Fluctuation (ALFF) and fractional ALFF (fALFF) """
    dfs = []
    for ses in args.seses:
        if 'gsr' in args.metrics[0]:
            fname = '_'.join([subj,ses])+'_task-rest_space-MNI152NLin2009cAsym_desc-detrend_gsr_smooth-6mm.nii.gz'
            bold_file = os.path.join(proj_dir, 'data/derivatives/post-fmriprep-fix/', subj, ses, 'func', fname)
        else:
            fname = '_'.join([subj,ses])+'_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            bold_file = os.path.join(proj_dir, 'data/derivatives/fmriprep-fix/', subj, ses, 'func', fname)

        stim_mask,stim_masker = get_subj_stim_mask(subj, args)
        if (stim_masker == None) :
            print("{} {} stimulus mask error".format(subj, ses))
            continue
        elif not(os.path.exists(bold_file)) :
            print(bold_file+" does not exists!")
            continue
        ts = stim_masker.fit()
        ts = stim_masker.transform_single_imgs(bold_file)

        freqs, Pxx = scipy.signal.welch(ts.squeeze(), fs=1./0.81, scaling='spectrum', nperseg=64, noverlap=32)
        if np.isnan(Pxx).any():
            print(subj +' PSD has NaNs, discard.')
            continue
        ALFF = np.sqrt(Pxx[(freqs >= 0.01) & (freqs <= 0.08)].mean())
        fALFF = ALFF / np.sqrt(Pxx[(freqs <= 0.25)].mean())

        dfs.append({'subj':subj, 'ses':ses, 'ALFF':ALFF, 'fALFF':fALFF}) #'stim_loc':np.array([l['x'], l['y'], l['z']]).flatten(),
        if args.verbose:
            print(subj + ' ' + ses + ' ALFF done.')
    return dfs


def plot_ALFF(df_summary, args):
    """ plot Amplitude Low Freq Fluctuations (ALFF) and Fractional ALFF """
    plt.figure(figsize=[20,10])
    plt.subplot(2,2,1)
    sbn.swarmplot(data=df_summary, x='group', y='fALFF', hue='ses', dodge=True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.subplot(2,2,2)
    sbn.pointplot(data=df_summary, x='ses', y='fALFF', hue='group', dodge=True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.subplot(2,2,3)
    sbn.swarmplot(data=df_summary, x='group', y='ALFF', hue='ses', dodge=True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.subplot(2,2,4)
    sbn.pointplot(data=df_summary, x='ses', y='ALFF', hue='group', dodge=True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    if args.save_figs:
        plt.savefig(os.path.join(proj_dir, 'img', 'ALFF_fALFF_stim_site_'+str(args.stim_radius)+'mm.svg'))



def compute_nbs(subjs, args):
    """ Network Based Statistics """
    g1=[]
    g2=[]
    for subj in subjs:
        group = get_group(subj)
        if group=='none':
            print(subj +' not in any group, discard.')
            continue
        fname = subj+'_ses-pre_task-rest_atlas-Schaefer2018_400_17+Tian_S4_desc-corr-detrend_filtered_scrub_gsr.h5'
        fpath = os.path.join('/home/sebastin/working/lab_lucac/shared/projects/ocd_clinical_trial/data/derivatives/post-fmriprep-fix/'+subj+'/ses-pre/fc', fname)
        if os.path.exists(fpath):
            with h5py.File(fpath, 'r') as f:
                pre = f['fc'][()]
        else:
            print(subj +' file not found, discard.')
            continue
        fname = subj+'_ses-post_task-rest_atlas-Schaefer2018_400_17+Tian_S4_desc-corr-detrend_filtered_scrub_gsr.h5'
        fpath = os.path.join('/home/sebastin/working/lab_lucac/shared/projects/ocd_clinical_trial/data/derivatives/post-fmriprep-fix/'+subj+'/ses-post/fc', fname)
        if os.path.exists(fpath):
            with h5py.File(fpath, 'r') as f:
                post = f['fc'][()]
        else:
            print(subj +' file not found, discard.')
            continue

        if args.nbs_session:
            g1.append(pre)
            g2.append(post)
        else:   #interaction
            if group=='group1':
                g1.append(pre-post)
            elif group=='group2':
                g2.append(pre-post)
    pvals, adj, null = bct.nbs_bct(np.array(g1).T, np.array(g2).T, thresh=args.nbs_thresh, paired=args.nbs_paired, k=args.n_perm, tail=args.nbs_tail)
    return pvals, adj, null


def get_kde(data, var, smoothing_factor=20, args=None):
    """ create kernel density estimate for the data (used in violin-like plots) """
    mn = pointplot_ylim[args.seed_type][var][0] # min
    mx = pointplot_ylim[args.seed_type][var][1] # max
    b = (mx-mn)/smoothing_factor
    model = KernelDensity(bandwidth=b)
    xtrain = np.array(data[var])[:, np.newaxis]
    model.fit(xtrain)
    xtest = np.linspace(mn,mx,100)[:, np.newaxis]
    log_dens = model.score_samples(xtest)
    mu = model.score_samples(np.array(data[var].mean()).reshape(1,1))
    return xtest, np.exp(log_dens), np.exp(mu)


def plot_pointplot(df_summary, args):
    """ Show indiviudal subject point plot for longitudinal display """
    plt.rcParams.update({'font.size': 16})
    df_summary = df_summary[df_summary['ses']!='pre-post']
    for i,var in enumerate(['corr', 'fALFF']):
        fig = plt.figure(figsize=[12,4])
        gs = plt.GridSpec(1,10)
        # ===========
        # point plots
        # ===========
        ax1 = fig.add_subplot(gs[0,1:4])
        ax1.set_ylim(pointplot_ylim[args.seed_type][var])
        ax1.set_xlim([-1, 1])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_yticklabels(labels=[], visible=False)
        ax1.set_ylabel('', visible=False)
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0,6:9])
        ax2.set_ylim(pointplot_ylim[args.seed_type][var])
        ax2.set_xlim([-0.1, 1.1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_yticklabels(labels=[], visible=False)
        ax2.set_ylabel('', visible=False)
        ax2.set_yticks([])

        for subj in df_summary.subj.unique():
            grp = str(df_summary[df_summary['subj']==subj].group.unique().squeeze())
            if grp == 'group1':
                plt.sca(ax2)
            elif grp == 'group2':
                plt.sca(ax1)
            else:
                continue
            sbn.pointplot(data=df_summary[df_summary['subj']==subj], x='ses', y=var, dodge=(np.random.rand()-0.5), color=group_colors[get_group(subj)], linewidth=0.5, alpha=0.5)

        plt.setp(ax1.lines, linewidth=0.75)
        plt.setp(ax1.collections, sizes=[10])
        plt.setp(ax2.lines, linewidth=0.75)
        plt.setp(ax2.collections, sizes=[10])
        if var=='corr':
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
            ax1.set_xticks([])
            ax1.spines['bottom'].set_visible(False)
            ax1.set_title('Sham')
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            ax2.set_xlabel('')
            ax2.spines['bottom'].set_visible(False)
            ax2.set_title('Active')
        else:
            ax1.set_xticklabels(['Baseline', 'post-cTBS'])
            #ax1.set_xlabel('Session')
            ax2.set_xticklabels(['Baseline', 'post-cTBS'])
            #ax2.set_xlabel('Session')


        # ============
        # violin plots
        # ============
        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_ylim(pointplot_ylim[args.seed_type][var])
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.set_xticklabels(labels=[], visible=False)
        ax0.set_xlabel('', visible=False)
        ax0.set_xticks([])
        data = df_summary[(df_summary.group=='group2') & (df_summary.ses=='ses-pre')]
        data = data[~data[var].isna()]
        x,y,mu = get_kde(data, var=var, args=args)
        ax0.fill(-y,x, color=group_colors['group2'], alpha=0.5)
        ax0.plot([-mu,0],[data[var].mean(), data[var].mean()], '-', color=group_colors['group2'])

        ax3 = fig.add_subplot(gs[0,4])
        ax3.set_ylim(pointplot_ylim[args.seed_type][var])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xticklabels(labels=[], visible=False)
        ax3.set_xlabel('', visible=False)
        ax3.set_xticks([])
        ax3.set_yticklabels(labels=[], visible=False)
        ax3.set_ylabel('', visible=False)
        ax3.set_yticks([])
        data = df_summary[(df_summary.group=='group2') & (df_summary.ses=='ses-post')]
        data = data[~data[var].isna()]
        x,y,mu = get_kde(data, var=var, args=args)
        ax3.fill(y,x, color=group_colors['group2'], alpha=0.5)
        ax3.plot([0,mu],[data[var].mean(), data[var].mean()], '-', color=group_colors['group2'])

        ax4 = fig.add_subplot(gs[0,5])
        ax4.set_ylim(pointplot_ylim[args.seed_type][var])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xticklabels(labels=[], visible=False)
        ax4.set_xlabel('', visible=False)
        ax4.set_xticks([])
        ax4.set_yticklabels(labels=[], visible=False)
        ax4.set_ylabel('', visible=False)
        ax4.set_yticks([])
        data = df_summary[(df_summary.group=='group1') & (df_summary.ses=='ses-pre')]
        data = data[~data[var].isna()]
        x,y,mu = get_kde(data, var=var, args=args)
        ax4.fill(-y,x, color=group_colors['group1'], alpha=0.5)
        ax4.plot([-mu,0],[data[var].mean(), data[var].mean()], '-', color=group_colors['group1'])

        ax5 = fig.add_subplot(gs[0,9])
        ax5.set_ylim(pointplot_ylim[args.seed_type][var])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xticklabels(labels=[], visible=False)
        ax5.set_xlabel('', visible=False)
        ax5.set_xticks([])
        ax5.set_yticklabels(labels=[], visible=False)
        ax5.set_ylabel('', visible=False)
        ax5.set_yticks([])
        data = df_summary[(df_summary.group=='group1') & (df_summary.ses=='ses-post')]
        data = data[~data[var].isna()]
        x,y,mu = get_kde(data, var=var, args=args)
        ax5.fill(y,x, color=group_colors['group1'], alpha=0.5)
        ax5.plot([0,mu],[data[var].mean(), data[var].mean()], '-', color=group_colors['group1'])

        plt.tight_layout()


        if args.save_figs:
            fname = '_'.join(['point_plot_distrib',args.metrics[0],var,
                            '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2)),datetime.now().strftime('%d%m%Y.pdf')])
            plt.savefig(os.path.join(proj_dir, 'img', fname))

        if args.plot_figs:
            plt.show(block=False)
        else:
            plt.close(fig)



def print_stats(df_summary, args):
    """ print stats of pre vs post variables """
    #df_summary.dropna(inplace=True)
    for var in ['corr', 'fALFF']:
        df = df_summary[~df_summary[var].isna()]
        df_pre = df[(df['ses']=='ses-pre')]
        df_post = df[(df['ses']=='ses-post')]

        diff_var = np.array(df_pre[var]) - np.array(df_post[var])
        diff_ybocs = np.array(df_pre['YBOCS_Total']) - np.array(df_post['YBOCS_Total'])

        r,p = scipy.stats.pearsonr(diff_var, diff_ybocs)
        print('Delta {}-YBOCS correlation across groups: r={:.2f}, p={:.3f}'.format(var,r,p))

        t,p = scipy.stats.ttest_ind(df_pre['YBOCS_Total'], df_post['YBOCS_Total'])
        print('YBOCS pre-post stats across groups: t={:.2f}, p={:.3f}'.format(t,p))

        for group in df.group.unique():
            t,p = scipy.stats.ttest_ind(np.array(df[(df['ses']=='ses-pre') & (df['group']==group)][var]), np.array(df[(df['ses']=='ses-post') & (df['group']==group)][var]) )
            print('{} pre-post {}  t={:.2f}  p={:.3f}'.format(var, group, t, p))

            df_pre = df[(df['ses']=='ses-pre') & (df['group']==group)]
            df_post = df[(df['ses']=='ses-post') & (df['group']==group)]

            diff_var = np.array(df_pre[var]) - np.array(df_post[var])
            diff_ybocs = np.array(df_pre['YBOCS_Total']) - np.array(df_post['YBOCS_Total'])

            r,p = scipy.stats.pearsonr(diff_var, diff_ybocs)
            print('Delta {}-YBOCS correlation in {}: r={:.2f}, p={:.3f}'.format(var,group,r,p))

            t,p = scipy.stats.ttest_ind(df_pre['YBOCS_Total'], df_post['YBOCS_Total'])
            print('YBOCS pre-post stats in {}: t={:.2f}, p={:.3f}'.format(group,t,p))

        print(var)

        diff_g1 = np.array(df[(df.ses=='ses-pre') & (df.group=='group1')][var]) - np.array(df[(df.ses=='ses-post') & (df.group=='group1')][var])
        diff_g2 = np.array(df[(df.ses=='ses-pre') & (df.group=='group2')][var]) - np.array(df[(df.ses=='ses-post') & (df.group=='group2')][var])
        print(pg.ttest(diff_g1, diff_g2, correction=False))

        mixed = pg.mixed_anova(data=df[df.ses!='pre-post'], dv=var, within='ses', between='group', subject='subj')
        pg.print_table(mixed)

        posthocs = pg.pairwise_ttests(data=df[df.ses!='pre-post'], dv=var, within='ses', between='group', subject='subj')
        pg.print_table(posthocs)






if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--seed_type', default='Harrison2009', type=str, action='store', help='choose Harrison2009, TianS4, etc')
    parser.add_argument('--atlas', default='Harrison2009', type=str, action='store', help='cortical and subcortical atlas, e.g. schaefer400_tianS4, etc')
    parser.add_argument('--compute_seed_corr', default=False, action='store_true', help="Flag to (re)compute seed to voxel correlations")
    parser.add_argument('--merge_LR_hemis', default=False, action='store_true', help="Flag to merge hemisphere's correlations")
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--subj', default=None, action='store', help='to process a single subject, give subject ID (default: process all subjects)')
    parser.add_argument('--use_gm_mask', default=False, action='store_true', help='use a whole brain gray matter mask to reduce the space of the second level analysis')
    parser.add_argument('--use_fspt_mask', default=False, action='store_true', help='use a fronto-striato-pallido-thalamic mask to reduce the space of the second level analysis')
    parser.add_argument('--use_cortical_mask', default=False, action='store_true', help='use a cortical gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_frontal_mask', default=False, action='store_true', help='use a frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_seed_specific_mask', default=False, action='store_true', help='use a seed-spefici frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_within_group_mask', default=False, action='store_true', help='use a union of within-group masks to reduce the space of the second level analysis')
    parser.add_argument('--unzip_corr_maps', default=False, action='store_true', help='unzip correlation maps for use in SPM (not necessary if only nilearn analysis)')
    parser.add_argument('--min_time_after_scrubbing', default=None, type=float, action='store', help='minimum time (in minutes) needed per subject needed to be part of the analysis (after scrubbing (None=keep all subjects))')
    parser.add_argument('--cluster_thresh', type=float, default=4., action='store', help="T stat to threshold to create clusters from voxel stats")
    parser.add_argument('--use_TFCE', default=False, action='store_true', help="use Threshold-Free Cluster Enhancement with randomise ")
    parser.add_argument('--OCD_minus_HC', default=False, action='store_true', help='direction of the t-test in FSL randomise -- default uses F-test')
    parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05, action='store', help="cluster level threshold, FDR corrected")
    parser.add_argument('--fpr_threshold', type=float, default=0.001, action='store', help="cluster level threshold, uncorrected")
    parser.add_argument('--within_group_threshold', type=float, default=0.005, action='store', help="threshold to create within-group masks")
    parser.add_argument('--compute_voi_corr', default=False, action='store_true', help="compute seed to VOI correlation and print stats")
    parser.add_argument('--two_sided_within_group', default=False, action='store_true', help="use two-tailed test to recreate within-group mask with parametric inference")
    parser.add_argument('--two_sided_between_group', default=False, action='store_true', help="use two-tailed test for between-group analysis with parametric inference")
    parser.add_argument('--n_perm', type=int, default=5000, action='store', help="number of permutation for non-parametric analysis")
    parser.add_argument('--within_mask_corr', default=False, action='store_true', help="compute FC within group masks and plot")
    parser.add_argument('--plot_within_group_masks', default=False, action='store_true', help="plot within-group masks used in second pass")
    parser.add_argument('--group_by_session', default=False, action='store_true', help="use a 4 columns design matrix with group by session interactions")
    parser.add_argument('--repeated2wayANOVA', default=False, action='store_true', help="use a n_1 + n_2 + 2 columns design with session and group by session interactions (2-way ANOVA with repeated measures)")
    parser.add_argument('--paired_design', default=False, action='store_true', help="makes diagonal design matrix")
    parser.add_argument('--stim_radius', type=float, default=5., action='store', help="radius of stim site assumed, centered at stim location")
    parser.add_argument('--compute_ALFF', default=False, action='store_true', help="compute Amplitude Low Frequency Fluctuation (ALFF) and fractional ALFF (fALFF)")
    parser.add_argument('--verbose', default=False, action='store_true', help="print out more processing info")
    parser.add_argument('--compute_t_contrasts', default=False, action='store_true', help="computes T contrasts in randomise")
    parser.add_argument('--compute_f_contrasts', default=False, action='store_true', help="computes F contrasts in randomise")
    parser.add_argument('--permuteBlocks', default=False, action='store_true', help="permute whole exchangability blocks rather than within blocks")
    parser.add_argument('--use_group_avg_stim_site', default=False, action='store_true', help='use a seed-spefici frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--compute_nbs', default=False, action='store_true', help="computes the network based statistics")
    parser.add_argument('--unilateral_seed', default=False, action='store_true', help="compute FC stats using only seed from one side (must be specified in header seed_loc)")
    parser.add_argument('--plot_pointplot', default=False, action='store_true', help="plot VOI correlation and fALFF pointplot (pre vs post)")
    parser.add_argument('--print_stats', default=False, action='store_true', help="print mixed ANOVA stats (group by session) and other stats (deltas YBOCS, FC, etc)")
    parser.add_argument('--nbs_session', default=False, action='store_true', help="perform NBS on session difference rather than the default interaction")
    parser.add_argument('--nbs_thresh', type=float, default=3.5, action='store', help="NBS stat threshold")
    parser.add_argument('--nbs_paired', default=False, action='store_true', help="NBS paired t-test")
    parser.add_argument('--nbs_tail', type=str, default='both', action='store', help="NBS t-test tail (both, right or left); default=both")
    args = parser.parse_args()

    if args.subj!=None:
        subjs = pd.Series([args.subj])
    else:
        subjs = pd.read_table(os.path.join(proj_dir, 'code', 'patients_list.txt'), names=['name'])['name']

    # options
    #atlases= ['Harrison2009'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4'] #schaefer400_harrison2009
    atlases = [args.atlas]
    pre_metric = 'seed_not_smoothed' #'unscrubbed_seed_not_smoothed'
    metrics =  ['detrend_gsr_filtered_scrubFD05'] #'detrend_gsr_smooth-6mm', 'detrend_gsr_filtered_scrubFD06'
    seses = ['ses-pre', 'ses-post']

    args.atlases = atlases
    args.pre_metric = pre_metric
    args.metrics = metrics
    args.seses = seses
    args.fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    args.in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/', args.seed_type, pre_metric)
    os.makedirs(args.in_dir, exist_ok=True)

    seeds, subrois = get_seed_names(args)
    if args.unilateral_seed:
        if args.seed_type=='Harrison2009':
            seeds = [seed for seed in seeds if seed[-1]=='R']
            subrois = seeds
        else:
            seeds = ['Right_'+seed for seed in seeds]
            subrois = seeds

    seedfunc = {'Harrison2009':sphere_seed_to_voxel,
            'TianS4':seed_to_voxel}

    # First remove subjects without enough data
    if args.min_time_after_scrubbing != None:
        subjs, revoked = get_subjs_after_scrubbing(subjs, seses, metrics, min_time=args.min_time_after_scrubbing)
    else:
        revoked=[]
    args.revoked=revoked

    # Then process data
    if args.compute_seed_corr:
        for atlas,ses in itertools.product(atlases,seses):
                if len(subjs)>1:
                    Parallel(n_jobs=args.n_jobs)(delayed(seedfunc[args.seed_type])(subj,ses,seeds,metrics,atlases,args) for subj in subjs)
                else:
                    seedfunc[args.seed_type](subjs.iloc[0],ses,seeds,metrics,atlases,args)

    if args.unzip_corr_maps:
        unzip_correlation_maps(subjs, seses, metrics, atlases, seeds, args)

    if args.merge_LR_hemis:
        in_fnames = merge_LR_hemis(subjs, subrois, seses, metrics, seed_type=str(seedfunc[args.seed_type]), args=args)

    if args.compute_voi_corr:
        df_voi_corr = compute_voi_corr(subjs, seeds=subrois, args=args)
        print_voi_stats(df_voi_corr, seeds=subrois, args=args)
        plot_voi_corr(df_voi_corr, seeds=subrois, args=args)

        if args.save_outputs:
            save_suffix = '_'.join([args.metrics[0],args.seed_type,args.fwhm])
            if args.unilateral_seed:
                save_suffix += '_unilateral'
            else:
                save_suffix += '_bilateral'
            if not args.use_group_avg_stim_site:
                save_suffix += '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2))
            with open(os.path.join(proj_dir, 'postprocessing', 'df_voi_corr_'+save_suffix+'.pkl'), 'wb') as f:
                pickle.dump(df_voi_corr,f)

    
    if args.compute_ALFF:
        if len(subjs) > 1:
            df_lines = Parallel(n_jobs=args.n_jobs, verbose=1)(delayed(compute_ALFF)(subj,args) for subj in subjs)
            df_lines = itertools.chain(*df_lines)
        else:
            df_lines = compute_ALFF(subjs[0], args)
        df_alff = pd.DataFrame(df_lines)
        df_summary = pd.merge(df_alff, df_groups)
        if args.plot_figs:
            plot_ALFF(df_summary, args)

        if args.save_outputs:
            save_suffix = '_'.join([args.metrics[0],args.seed_type,args.fwhm])
            if not args.use_group_avg_stim_site:
                save_suffix += '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2))
            with open(os.path.join(proj_dir, 'postprocessing', 'df_alff_'+save_suffix+'.pkl'), 'wb') as f:
                pickle.dump(df_summary,f)

    if args.compute_nbs:
        out_nbs = compute_nbs(subjs, args)
        if args.save_outputs:
            save_suffix = '_10thr{}'.format(int(args.nbs_thresh*10))
            if args.nbs_session:
                save_suffix += '_session'
            else:
                save_suffix += '_interaction'
            if args.nbs_paired:
                save_suffix += '_paired'
            save_suffix += '_{}_tail_{}perms'.format(args.nbs_tail, args.n_perm)
            today = datetime.now().strftime("%Y%m%d")
            save_suffix += '_'+today
            with open(os.path.join(proj_dir, 'postprocessing', 'nbs'+save_suffix+'.pkl'), 'wb') as f:
                pickle.dump(out_nbs,f)

    if args.plot_pointplot:
        # loadings
        if (('df_alff' not in locals()) & ('df_alff' not in globals())):
            save_suffix = '_'.join([metrics[0],args.seed_type,args.fwhm])
            if not args.use_group_avg_stim_site:
                save_suffix += '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2))
            with open(os.path.join(proj_dir, 'postprocessing', 'df_alff_'+save_suffix+'.pkl'), 'rb') as f:
                df_alff = pickle.load(f)
        if (('df_voi_corr' not in locals()) & ('df_voi_corr' not in globals())):
            save_suffix = '_'.join([metrics[0],args.seed_type,args.fwhm])
            if args.unilateral_seed:
                save_suffix += '_unilateral'
            else:
                save_suffix += '_bilateral'
            if not args.use_group_avg_stim_site:
                save_suffix += '_indStimSite_{}mm_diameter'.format(int(args.stim_radius*2))
            with open(os.path.join(proj_dir, 'postprocessing', 'df_voi_corr_'+save_suffix+'.pkl'), 'rb') as f:
                df_voi_corr = pickle.load(f)
        with open(os.path.join(proj_dir, 'postprocessing', 'df_pat.pkl'), 'rb') as f:
            df_pat = pickle.load(f)
        df_summary = df_alff.merge(df_voi_corr).merge(df_pat)
        # plotting
        plot_pointplot(df_summary, args)

    # stats
    if args.print_stats:
        print_stats(df_summary, args)

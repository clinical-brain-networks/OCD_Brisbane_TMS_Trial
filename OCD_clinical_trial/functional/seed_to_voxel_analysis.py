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


def create_design_matrix(subjs, args):
    """ Create a more complex design matrix with group by hemisphere interactions """
    n_1 = np.sum(['group1' in get_group(s) for s in subjs if s not in args.revoked])
    n_2 = np.sum(['group2' in get_group(s) for s in subjs if s not in args.revoked])
    if args.group_by_session:
      if args.paired_design:
        design_mat = np.eye((2*n_1+2*n_2))
        design_matrix = pd.DataFrame(design_mat)
      elif args.repeated2wayANOVA:
        design_mat = np.zeros((2*n_1+2*n_2, n_1+n_2+2), dtype=int)
        # subject means :
        design_mat[:n_1,:n_1] = np.eye(n_1)
        design_mat[n_1:2*n_1,:n_1] = np.eye(n_1)
        design_mat[2*n_1:2*n_1+n_2,n_1:n_1+n_2] = np.eye(n_2)
        design_mat[2*n_1+n_2:,n_1:n_1+n_2] = np.eye(n_2)
        # session main effect:
        design_mat[:n_1,-2] = 1
        design_mat[n_1:2*n_1,-2] = -1
        design_mat[-2*n_2:-n_2,-2] = 1
        design_mat[-n_2:,-2] = -1
        # group by session interaction
        design_mat[:n_1,-1] = 1
        design_mat[n_1:2*n_1,-1] = -1
        design_mat[-2*n_2:-n_2,-1] = -1
        design_mat[-n_2:,-1] = 1

        design_matrix = pd.DataFrame(design_mat)
      else:
        design_mat = np.zeros((2*(n_1+n_2),4), dtype=int)

        design_mat[:n_1, 0] = 1 # group1_pre
        design_mat[n_1:2*n_1, 1] = 1 # group1_post
        design_mat[-2*n_2:-n_2, 2] = 1 # group2_pre
        design_mat[-n_2:, 3] = 1 # group2_post

        design_matrix = pd.DataFrame()
        design_matrix['group1_pre'] = design_mat[:,0]
        design_matrix['group1_post'] = design_mat[:,1]
        design_matrix['group2_pre'] = design_mat[:,2]
        design_matrix['group2_post'] = design_mat[:,3]
    else:
        design_mat = np.zeros((n_1+n_2,2), dtype=int)
        design_mat[:n_1,0] = 1
        design_mat[-n_2:, 1] = 1

        design_matrix = pd.DataFrame()
        design_matrix['group1'] = design_mat[:,0]
        design_matrix['group2'] = design_mat[:,1]
    return design_matrix


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
            # prepare output file
            #hfname = subj+'_task-rest_'+atlas+'_desc-'+metric+'_'+''.join(seeds)+'_seeds_ts.h5'
            #hf = h5py.File(os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'timeseries', hfname), 'w')

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
                #hf.create_dataset(seed+'_ts', data=seed_ts)
            #hf.close()
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


def prep_fsl_randomise(in_fnames, seeds, metrics, args):
    """ prepare 4D images for FSL randomise """
    gm_mask = datasets.load_mni152_gm_mask()
    masker = NiftiMasker(gm_mask)

    ind_mask = False  # whether to apply GM mask to individual 3D images separatetely or already combined in 4D image
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    for metric,seed in itertools.product(metrics,seeds):
        if ind_mask:
            imgs = None
            for in_fname in in_fnames[(seed,metric)]:
                img = load_img(in_fname)
                masker.fit(img)
                masker.generate_report()
                masked_data = masker.transform(img)
                masked_img = masker.inverse_transform(masked_data)
                if imgs==None:
                    imgs = img
                else:
                    imgs = nilearn.image.concat_imgs([imgs, masked_img], auto_resample=True)
        else:
            img = nilearn.image.concat_imgs(in_fnames[(seed,metric)], auto_resample=True, verbose=0)
            masker.fit(img)
            masker.generate_report()
            masked_data = masker.transform(img)
            imgs = masker.inverse_transform(masked_data) #nib.Nifti1Image(masked_data, img.affine, img.header)
        nib.save(imgs, os.path.join(args.in_dir, metric, fwhm, 'masked_resampled_pairedT4D_'+seed))

    # saving design matrix and contrast
    design_matrix = create_design_matrix(subjs, args)
    design_con = np.hstack((1, -1)).astype(int)
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_mat'), design_matrix, fmt='%i')
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_con'), design_con, fmt='%i', newline=' ')


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


def create_local_sphere_within_cluster(vois, rois, metrics, args=None, sphere_radius=3.5):
    """ create a sphere VOIs of given radius within cluster VOIs (for DCM analysis) """
    max_locals = dict( ( ( (roi,metric) , {'controls':[], 'patients':[]}) for roi,metric in itertools.product(rois, metrics) ) )
    for metric in metrics:
        for roi,voi in zip(roi,vois):
            for subj in subjs:
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                corr_file = '_'.join([subj,metric,fwhm,atlas,roi,'ns_sphere_roi_to_voxel_corr.nii'])
                corr_fname = os.path.join(corr_path, roi, coh, corr_file)
                img = load_img(corr_fname)
                mask = load_img(baseline_dir+'/postprocessing/SPM/rois_and_rois/'+voi+'.nii')
                mask = nilearn.image.new_img_like(img, mask.get_fdata())
                new_data = np.abs(img.get_fdata()) * mask.get_fdata()
                local_max = new_data.max()
                max_coords = np.where(new_data>=local_max)
                local_max_coords = nilearn.image.coord_transform(max_coords[0], max_coords[1], max_coords[2], img.affine)
                new_mask = nltools.create_sphere(local_max_coords, radius=sphere_radius)
                out_path = os.path.join(baseline_dir, 'postprocessing', subj, 'spm', 'masks')
                os.makedirs(out_path, exist_ok=True)
                fname = os.path.join(out_path, 'local_'+voi+'_'+metric+'.nii')
                nib.save(new_mask, fname)
                max_locals[roi,metric][coh].append(new_mask)
    return max_locals


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

def perform_second_level_analysis(seed, metric, design_matrix, cohorts=['controls', 'patients'], args=None, masks=[]):
    """ Perform second level analysis based on seed-to-voxel correlation maps """
    # naming convention in file system
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    # get images path
    con_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'patients', '*'))
    flist = np.hstack([con_flist, pat_flist])

    # remove revoked subjects
    if args.revoked != []:
        flist = [l for l in flist if ~np.any([s in l for s in revoked])]

    imgs, masker, mask = mask_imgs(flist, masks=masks, seed=seed, args=args)

    # perform analysis
    t_glm = time()
    glm = SecondLevelModel(mask_img=masker)
    glm.fit(imgs, design_matrix=design_matrix)
    print('GLM fitting took {:.2f}s'.format(time()-t_glm))

    contrasts = dict()
    t0 = time()
    contrasts['within_con'] = glm.compute_contrast(np.array([1, 0]), output_type='all')
    t1 = time()
    contrasts['within_pat'] = glm.compute_contrast(np.array([0, 1]), output_type='all')
    t2 =  time()
    contrasts['between'] = glm.compute_contrast(np.array([1, -1]), output_type='all')
    print('within groups and between group contrasts took {:.2f}, {:.2f} and {:.2f}s'.format(t1-t0, t2-t1, time()-t2))
    n_voxels = np.sum(nilearn.image.get_data(glm.masker_.mask_img_))
    params = glm.get_params()
    return contrasts, n_voxels, params

def threshold_contrast(contrast, height_control='fpr', alpha=0.005, cluster_threshold=10):
    """ cluster threshold contrast at alpha with height_control method for multiple comparisons """
    thresholded_img, thresh = threshold_stats_img(
        contrast, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
    cluster_table = get_clusters_table(
        contrast, stat_threshold=thresh, cluster_threshold=cluster_threshold,
        two_sided=True, min_distance=5.0)
    return thresholded_img, thresh, cluster_table

def create_within_group_mask(subroi_glm_results, args):
    """ create within group masks to use for between group contrasts to improve SNR """
    con_img, con_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_con']['z_score'],
                                    cluster_threshold=100, alpha=args.within_group_threshold)
    con_mask = binarize_img(con_img, threshold=con_thr)
    pat_img, pat_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_pat']['z_score'],
                                    cluster_threshold=100, alpha=args.within_group_threshold)
    pat_mask = binarize_img(pat_img, threshold=pat_thr)
    mask = nilearn.masking.intersect_masks([con_mask, pat_mask], threshold=1, connected=False) # thr=1: intersection; thr=0: union
    return mask, con_mask, pat_mask


def run_second_level(subjs, metrics, subrois, args):
    """ Run second level analysis """
    design_matrix = create_design_matrix(subjs, args)

    glm_results = dict()
    for metric,subroi in itertools.product(metrics,subrois):
        print('Starting 2nd level analysis for '+subroi+' subroi.')
        t0 = time()
        glm_results[subroi] = dict()

        t_fp = time()
        contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args, masks=[])
        glm_results[subroi]['first_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['first_pass','params'] = contrasts, n_voxels, params
        print('{} first pass in {:.2f}s'.format(subroi,time()-t_fp))

        passes = ['first_pass']

        if args.use_within_group_mask:
            t_wmask = time()
            within_group_mask, con_mask, pat_mask = create_within_group_mask(glm_results[subroi], args)
            glm_results[subroi]['within_group_mask'], glm_results[subroi]['con_mask'], glm_results[subroi]['pat_mask'] = within_group_mask, con_mask, pat_mask
            print('created within groups mask in {:.2f}s'.format(time()-t_wmask))

            t_sp = time()
            contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args, masks=[within_group_mask])
            glm_results[subroi]['second_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['second_pass','params'] = contrasts, n_voxels, params
            print('{} second pass in {:.2f}s'.format(subroi,time()-t_sp))

            passes.append('second_pass')

        # Correcting the p-values for multiple testing and taking negative logarithm
        #neg_log_pval = nilearn.image.math_img("-np.log10(np.maximum(1, img * {}))"
        #                .format(str(glm_results[subroi]['n_voxels'])),
        #                img=glm_results[subroi]['contrasts']['between']['p_value'])
        #glm_results[subroi]['neg_log_pval'] = neg_log_pval

        t_thr = time()
        for pss in passes:
            glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresholded_img')], \
                glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')], \
                glm_results[subroi][(pss,'fpr',args.fpr_threshold,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'])

            print(' '.join([subroi,pss,'clusters at p<{:.3f} uncorrected:'.format(args.fpr_threshold)]))
            print(glm_results[subroi][(pss,'fpr',args.fpr_threshold,'cluster_table')])

            glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresholded_img')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'], height_control='fdr', alpha=args.fdr_threshold)

            print(' '.join([subroi,pss,'clusters at p<{:.2f} FDR corrected:'.format(args.fdr_threshold)]))
            print(glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')])

        print('Thresholding and clustering took {:.2f}s'.format(time()-t_thr))

        if args.plot_figs:
            t_plt = time()
            for pss in passes:
                fig = plt.figure(figsize=[16,4])
                ax1 = plt.subplot(1,2,1)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')],
                                axes=ax1, title='_'.join([pss,subroi,'contrast_fpr'+str(args.fpr_threshold)]))
                ax2 = plt.subplot(1,2,2)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')],
                                axes=ax2, title='_'.join([pss,subroi,'contrast_fdr'+str(args.fdr_threshold)]))
                print('{} plotting took {:.2f}s'.format(subroi,time()-t_plt))

                if args.save_figs:
                    plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')],
                    output_file=os.path.join(args.out_dir,subroi+'_'+pss+'_contrast_fpr{:.3f}.pdf'.format(args.fpr_threshold)))

        print('Finished 2nd level analysis for '+subroi+' ROI in {:.2f}s'.format(time()-t0))

    # savings
    if args.save_outputs:
        suffix = '_'+metric
        if args.min_time_after_scrubbing!=None:
            suffix += '_minLength'+str(int(args.min_time_after_scrubbing*10))
        if args.use_fspt_mask:
            suffix += '_fsptMask'
        if args.use_cortical_mask:
            suffix += '_corticalMask'
        today = datetime.datetime.now().strftime("%Y%m%d")
        suffix += '_'+today
        with gzip.open(os.path.join(args.out_dir,'glm_results'+suffix+'.pkl.gz'), 'wb') as of:
            pickle.dump(glm_results, of)

    return glm_results


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
    #df_voi_corr['corr'] = df_voi_corr['corr'] / 880.
    #df_voi_corr['corr'].loc[df_voi_corr['corr']>1] = 1
    #df_voi_corr['corr'].loc[df_voi_corr['corr']<-1] = -1

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


def compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, seed, args):
    """ reconstruct within-group masks using non-parametric inference """
    # controls
    imgs, masker, mask = mask_imgs(con_flist, masks=masks, seed=seed, args=args)
    neg_log_pvals_within_con = non_parametric_inference(list(np.sort(con_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(con_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_con_mask = binarize_img(neg_log_pvals_within_con, threshold=0.) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    # patients
    imgs, masker, mask = mask_imgs(pat_flist, masks=masks, seed=seed, args=args)
    neg_log_pvals_within_pat = non_parametric_inference(list(np.sort(pat_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(pat_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_pat_mask = binarize_img(neg_log_pvals_within_pat, threshold=0.) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    within_groups_mask = nilearn.masking.intersect_masks([within_con_mask, within_pat_mask], threshold=0, connected=False) # thr=0:union, thr=1:intersection
    return within_groups_mask


def non_parametric_analysis(subjs, seed, metric, pre_metric, masks=[], args=None):
    """ Performs a non-parametric inference """
    post_metric = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    in_dir = os.path.join(baseline_dir, 'postprocessing/SPM/input_imgs/', args.seed_type, pre_metric, metric, post_metric)

    # create imgs file list
    con_flist = glob.glob(os.path.join(in_dir, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(in_dir, seed, 'patients', '*'))
    pat_flist = [pf for pf in pat_flist if 'sub-patient16' not in pf]
    flist = np.hstack([con_flist, pat_flist])

    design_matrix = create_design_matrix(subjs, args)

    if args.use_SPM_mask:
        # template masking (& SPM within-group masks) need to be set manually via commenting others
        mask = os.path.join(baseline_dir, 'postprocessing/SPM/outputs/',args.seed_type, 'smoothed_but_sphere_seed_based', metric, seed, 'scrub_out_1/'+seed+'_fl_0001unc_005fwe.nii.gz') # within-group
        #fspt_mask = binarize_img(os.path.join(baseline_dir, 'utils', 'Larger_FrStrPalThal_schaefer100_tianS1MNI_lps_mni.nii.gz')) # fronto-striato-pallido-thalamic (fspt) mask
        #gm_mask = os.path.join(baseline_dir, 'utils', 'schaefer_cortical.nii.gz') # gray matter only
    elif args.use_within_group_mask:
        # create within-group masks non-parametrically
        mask = compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, seed, args)
    else:
        imgs, masker, mask = mask_imgs(flist, masks=masks, seed=seed, args=args)

    # between-groups non-parametric inference
    neg_log_pvals_permuted_ols_unmasked = \
        non_parametric_inference(list(np.sort(flist)),
                                 design_matrix=pd.DataFrame(design_matrix['con'] - design_matrix['pat']),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_between_group, mask=mask, n_jobs=10, verbose=1)

    return neg_log_pvals_permuted_ols_unmasked, mask

def plot_non_param_maps(neg_log_pvals, seed, suffix, args=None):
    """ plot and save outputs of the non-parametric analysis """
    if args.plot_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False) #cut_coords=[-24,55,34]
    if args.save_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False,
                      output_file=os.path.join(baseline_dir, 'img', '_'.join([seed,suffix])+'.pdf')) #cut_coords=[-24,55,34]


def compute_FC_within_masks(subjs, np_results, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ compute FC within masks used for the between-group analysis """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            if 'control' in subj:
                cohort = 'controls'
            else:
                cohort = 'patients'
            for seed in seeds:
                # load correlation map
                fname = '_'.join([subj, metric, fwhm, atlas, seed, 'ns_sphere_seed_to_voxel_corr.nii'])
                corr_map = load_img(os.path.join(baseline_dir, 'postprocessing/SPM/input_imgs/', args.seewd_type, 'seed_not_smoothed',
                                    metric, fwhm, seed, cohort, fname))
                voi_mask = resample_to_img(np_results[seed]['mask'], corr_map, interpolation='nearest')

                # extract correlations
                voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                for corr in np.ravel(voi_corr[voi_corr!=0]):
                    df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'seed':seed, 'corr':corr}
                    dfs.append(df_line)
    df_mask_corr = pd.DataFrame(dfs)
    return df_mask_corr

def plot_within_mask_corr(df_mask_corr, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ bar plots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.15, 0.3]
    fig = plt.figure(figsize=[12,6])
    df_mask_corr['corr'] = df_mask_corr['corr'] / 880.
    df_mask_corr['corr'].loc[df_mask_corr['corr']>1] = 1
    df_mask_corr['corr'].loc[df_mask_corr['corr']<-1] = -1

    for i,seed in enumerate(seeds):
      ax = plt.subplot(1,len(seeds),i+1)
      #sbn.barplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v')
      #sbn.swarmplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', dodge=True, size=0.1)
      sbn.violinplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', split=True, scale_hue=True,
                     inner='quartile', dodge=True, width=0.8, cut=1)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      if i==len(seeds)-1:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      else:
        ax.get_legend().set_visible(False)
      plt.tight_layout()

    if args.save_figs:
        figname = 'seed_to_mask_corr_3seeds.svg'
        plt.savefig(os.path.join(baseline_dir, 'img', figname))


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


def create_contrast_vector(subjs, args):
    """ create contrast vector based on options given in arguments (default: only group difference) """
    suffix = ''
    n_1 = np.sum(['group1' in get_group(s) for s in subjs if s not in args.revoked])
    n_2 = np.sum(['group2' in get_group(s) for s in subjs if s not in args.revoked])
    if args.group_by_session:
        suffix += '_group_by_session'
        if args.paired_design:
          cv = []
          #cv.append(np.ones([1,1+n_1+n_2]))
          #cv.append(np.concatenate([-np.ones([1,1]).ravel(), np.ones([1,n_1+n_2]).ravel()]))
          #cv.append(np.concatenate([-np.ones([1,1]).ravel(), -np.ones([1,n_1+n_2]).ravel()]))
          #cv.append(np.concatenate([np.ones([1,1]).ravel(), -np.ones([1,n_1+n_2]).ravel()]))
          cv.append(np.concatenate([np.ones((1,2*n_1)).ravel(), -np.ones((1,2*n_2)).ravel()]))
          cv.append(np.concatenate([np.ones((1,n_1)).ravel(), -np.ones((1,n_1)).ravel(), np.ones((1,n_2)).ravel(), -np.ones((1,n_2)).ravel()]))
          cv.append(np.concatenate([np.ones((1,n_1)).ravel(), -np.ones((1,n_1)).ravel(), -np.ones((1,n_2)).ravel(), np.ones((1,n_2)).ravel()]))
          cv = np.array(cv)
          suffix += '_paired'
        elif args.repeated2wayANOVA:
          cv = np.zeros( (6, n_1 + n_2 + 2) ) # 3 contrasts (one positive + one negative each)
          # group main effect
          cv[0,:n_1] = 1
          cv[0,n_1:n_1+n_2] = -1
          cv[1,:n_1] = -1
          cv[1,n_1:n_1+n_2] = 1
          # session  main effect
          cv[2,-2] = 1
          cv[3,-2] = -1
          # interactions
          cv[4,-1] = 1
          cv[5,-1] = -1
          # F test
          cm = np.array([[1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1]])
          #grp = np.ones((2*n_1+2*n_2,1))
          grp = np.concatenate([np.arange(n_1), np.arange(n_1), np.arange(n_1,n_1+n_2), np.arange(n_1,n_1+n_2)])+1 # offset of 1 because not sure what 0 would d
          suffix += '_grp123_repeated2wayANOVA'
        else:
            cv = np.array([[1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1], [1,-1,0,0], [0,0,1,-1]])
            cm = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,-1]])
            #grp = np.ones((2*n_1+2*n_2,1))
            grp = np.concatenate([np.arange(n_1), np.arange(n_1), np.arange(n_1,n_1+n_2), np.arange(n_1,n_1+n_2)])+1 # offset of 1 because not sure what 0 would d
        suffix += '_Ftest'
    else:
        if args.OCD_minus_HC:
            cv = np.array([[-1, 1]])
            suffix += '_group2_minus_group1'
            con_type = 't'
        else:
            cv = np.array([[1, -1]])
            suffix += '_group1_minus_group2'
            con_type = 't'
        grp = np.ones((n_1+n_2,1))
        cm = np.array([[1]])
        suffix += '_Ftest'
    return cv.astype(int), cm.astype(int), grp.astype(int), suffix

def get_final_suffix(dcon, dfts, dgrp, mask_file, suffix, args):
    """ create suffix (options) for the FSL randomise command and its output file """
    cmd_suffix = ' ' # start with suffix
    file_suffix = suffix+'_outputs_n'+str(args.n_perm)
    if args.compute_t_contrasts:
        cmd_suffix += (' -t '+dcon)
    if args.compute_f_contrasts:
        cmd_suffix += (' -f '+dfts)
    cmd_suffix += (' -e '+dgrp+' -m '+mask_file+' -n '+str(args.n_perm))
    if args.permuteBlocks:
        cmd_suffix += ' --permuteBlocks'
        file_suffix += '_permuteBlocks'
    if args.use_TFCE:
        cmd_suffix += (' -T --uncorrp')
    else:
        cmd_suffix += (' -c '+str(args.cluster_thresh)+' --uncorrp')
        file_suffix += '_c'+str(int(args.cluster_thresh*10))
    file_suffix += datetime.now().strftime('_%d%m%Y')
    return cmd_suffix, file_suffix



def use_randomise(subjs, seed, atlas, metric, args=None):
    """ perform non-parametric inference using FSL randomise and cluster-based enhancement """
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    _,_,flist = get_file_lists(subjs, seed, atlas, metric, args)

    # create 4D image from list of 3D
    imgs_4D = nilearn.image.concat_imgs(flist, auto_resample=True)

    dm = create_design_matrix(subjs, args)
    cv, cm, grp, suffix = create_contrast_vector(subjs, args)

    # outputs/savings to file
    out_dir = os.path.join(proj_dir, 'postprocessing/SPM/outputs/', args.seed_type, 'smoothed_but_sphere_seed_based', metric, fwhm, seed, 'randomise')
    os.makedirs(out_dir, exist_ok=True)

    # set mask
    mask_file = os.path.join(out_dir, seed+'_pathway_mask'+suffix+'.nii.gz')
    _,_,mask = mask_imgs(flist, seed=seed, masks=[],  args=args)
    mask = resample_to_img(mask, imgs_4D, interpolation='nearest')
    nib.save(mask, mask_file)

    # set paths for design, t contrasts, f contrasts, and exchange block files
    dmat = os.path.join(out_dir, 'design.mat')
    dcon = os.path.join(out_dir, 'design.con')
    dfts = os.path.join(out_dir, 'design.fts')
    dgrp = os.path.join(out_dir, 'design.grp')

    cmd_suffix, file_suffix = get_final_suffix(dcon, dfts, dgrp, mask_file, suffix, args)

    dm.to_csv(os.path.join(out_dir, 'design_mat'+file_suffix), sep=' ', index=False, header=False)
    np.savetxt(os.path.join(out_dir, 'design_con'+file_suffix), cv, fmt='%i')
    np.savetxt(os.path.join(out_dir, 'design_fts'+file_suffix), cm, fmt='%i')
    np.savetxt(os.path.join(out_dir, 'design_grp'+file_suffix), grp, fmt='%i')

    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_mat'+file_suffix), dmat))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_con'+file_suffix), dcon))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_fts'+file_suffix), dfts))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_grp'+file_suffix), dgrp))

    in_file = os.path.join(out_dir, seed+'_imgs_4D'+file_suffix+'.nii.gz')
    nib.save(imgs_4D, in_file)

    out_file = os.path.join(out_dir, seed+file_suffix)
    cmd = 'randomise -i '+in_file+' -o '+out_file+' -d '+dmat + cmd_suffix

    print(cmd)
    #pdb.set_trace()
    os.system(cmd)
    os.remove(in_file)


def plot_randomise_outputs(subjs, seed, metric, args, stat='t'):
    """ plot the outcomes of the non-paramteric infernece using randomise and TFCE """
    locs = {'Acc':None,
            'dCaud':None,
            'dPut':None,
            'vPut':[-49,30,12],
            'NucleusAccumbens':None}
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    cv,cm,grp,suffix = create_contrast_vector(subjs, args)

    #suffix += '_noExBlocks_07092022'#+datetime.now().strftime('%d%m%Y')
    out_dir = os.path.join(proj_dir, 'postprocessing/SPM/outputs/', args.seed_type, 'smoothed_but_sphere_seed_based', metric, fwhm, seed, 'randomise')

    suffix += '_outputs_n'+str(args.n_perm)
    if args.permuteBlocks:
        suffix += '_permuteBlocks'
    if not args.use_TFCE:
        suffix += '_c'+str(int(args.cluster_thresh*10))
    suffix += datetime.now().strftime('_%d%m%Y')

    for i in np.arange(1,4):

        plt.figure(figsize=[16,12])

        # FWE p-values
        out_file = os.path.join(out_dir, seed+suffix+'_tfce_corrp_'+stat+'stat{}.nii.gz'.format(i))
        ax1 = plt.subplot(3,2,1)
        plot_stat_map(out_file, axes=ax1, draw_cross=False, title=seed+' randomise -- corrp {}stat{}'.format(stat,i))
        ax2 = plt.subplot(3,2,2)
        plot_stat_map(out_file, threshold=0.95, axes=ax2, draw_cross=False, cmap='Oranges',
                        title=seed+' randomise -- corrp>0.95 (p<0.05)',
                        cut_coords=locs[seed])

        # stats
        out_file = os.path.join(out_dir, seed+suffix+'_{}stat{}.nii.gz'.format(stat,i))
        ax3 = plt.subplot(3,2,3)
        plot_stat_map(out_file, axes=ax3, draw_cross=False, title=seed+' randomise -- {}stat{}'.format(stat,i))
        ax4 = plt.subplot(3,2,4)
        plot_stat_map(out_file, threshold=args.cluster_thresh, axes=ax4, draw_cross=False, cmap='Oranges', title=seed+' randomise -- {}stat{}>{:.1f}'.format(stat,i,args.cluster_thresh))

        # FDR p-vals
        out_file = os.path.join(out_dir, seed+suffix+'_tfce_p_{}stat{}.nii.gz'.format(stat,i))
        ax5 = plt.subplot(3,2,5)
        plot_stat_map(out_file, axes=ax5, draw_cross=False, title=seed+' p_unc '+stat+'stat'+str(i))
        ax6 = plt.subplot(3,2,6)
        plot_stat_map(out_file, threshold=0.999, axes=ax6, draw_cross=False, cmap='Oranges', title=seed+' p_unc<0.001 '+stat+'stat'+str(i))


def plot_within_group_masks(subrois, glm_results, args):
    """ display maps of within-group contrasts """
    for subroi in subrois:
        plt.figure(figsize=[18,4])
        ax1 = plt.subplot(1,3,1)
        plot_stat_map(glm_results[subroi]['con_mask'],
                    axes=ax1, title=subroi+' within_con p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)
        ax2 = plt.subplot(1,3,2)
        plot_stat_map(glm_results[subroi]['pat_mask'],
                    axes=ax2, title=subroi+' within_pat p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)
        ax3 = plt.subplot(1,3,3)
        plot_stat_map(glm_results[subroi]['within_group_mask'],
                    axes=ax3, title=subroi+' within-group mask p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)



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
        #stim_mask = resample_to_img(stim_mask, bold_file, interpolation='nearest')
        #ts = nilearn.masking.apply_mask(bold_file, stim_mask, smoothing_fwhm=args.brain_smoothing_fwhm)

        freqs, Pxx = scipy.signal.welch(ts.squeeze(), fs=1./0.81, scaling='spectrum', nperseg=64, noverlap=32)
        #freqs, Pxx = scipy.signal.periodogram(ts.squeeze().astype(float), fs=1./0.81, scaling='spectrum', detrend=False, window='hann')
        #Pxx, freqs = plt.psd(ts.squeeze(), Fs=1./0.81);
        #Pxx = np.sqrt(Pxx)
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


def compute_diff_ALFF(df_alff, args):
    """ t-test on individual's ALFF at sim site """
    return

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

        """
        for i,group in enumerate(['group2', 'group1']):
            if group=='group1':
                plt.sca(ax4)
                ax = plt.gca()
            else:
                plt.sca(ax3)
                ax = plt.gca()
            sbn.violinplot(data=df_summary[df_summary.group==group], x='group', y=var, hue='ses', dodge=True, split=True, palette={'ses-pre':group_colors[group], 'ses-post':group_colors[group]}, linewidth=1, inner='quartile', scale='width')
            plt.setp(ax.collections, alpha=.5)
            plt.legend().set_visible(False)
        """
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
    parser.add_argument('--run_second_level', default=False, action='store_true', help='run second level statistics')
    parser.add_argument('--use_gm_mask', default=False, action='store_true', help='use a whole brain gray matter mask to reduce the space of the second level analysis')
    parser.add_argument('--use_fspt_mask', default=False, action='store_true', help='use a fronto-striato-pallido-thalamic mask to reduce the space of the second level analysis')
    parser.add_argument('--use_cortical_mask', default=False, action='store_true', help='use a cortical gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_frontal_mask', default=False, action='store_true', help='use a frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_seed_specific_mask', default=False, action='store_true', help='use a seed-spefici frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_within_group_mask', default=False, action='store_true', help='use a union of within-group masks to reduce the space of the second level analysis')
    parser.add_argument('--unzip_corr_maps', default=False, action='store_true', help='unzip correlation maps for use in SPM (not necessary if only nilearn analysis)')
    parser.add_argument('--min_time_after_scrubbing', default=None, type=float, action='store', help='minimum time (in minutes) needed per subject needed to be part of the analysis (after scrubbing (None=keep all subjects))')
    parser.add_argument('--prep_fsl_randomise', default=False, action='store_true', help='Prepare 4D images for running FSL randomise')
    parser.add_argument('--use_randomise', default=False, action='store_true', help='run FSL randomise -- independent from prep_fsl_randomise')
    parser.add_argument('--cluster_thresh', type=float, default=4., action='store', help="T stat to threshold to create clusters from voxel stats")
    parser.add_argument('--use_TFCE', default=False, action='store_true', help="use Threshold-Free Cluster Enhancement with randomise ")
    parser.add_argument('--OCD_minus_HC', default=False, action='store_true', help='direction of the t-test in FSL randomise -- default uses F-test')
    parser.add_argument('--create_sphere_within_cluster', default=False, action='store_true', help='export sphere around peak within VOI cluster in prep for DCM analysis')
    parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05, action='store', help="cluster level threshold, FDR corrected")
    parser.add_argument('--fpr_threshold', type=float, default=0.001, action='store', help="cluster level threshold, uncorrected")
    parser.add_argument('--within_group_threshold', type=float, default=0.005, action='store', help="threshold to create within-group masks")
    parser.add_argument('--compute_voi_corr', default=False, action='store_true', help="compute seed to VOI correlation and print stats")
    parser.add_argument('--non_parametric_analysis', default=False, action='store_true', help="compute between group analysis using non-parametric inference")
    parser.add_argument('--use_SPM_mask', default=False, action='store_true', help="use within-group masks generated from SPM (one-tailed)")
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
    #subjs = pd.Series(['sub-patient04', 'sub-patient54'])

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

    #seeds = list(seed_loc.keys()) #['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']
    #subrois = np.unique([seed[:-1] for seed in seeds])#['Acc', 'dCaud', 'dPut', 'vPut', 'drPut']
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

        # can only prep fsl with in_fnames from merge_LR_hemis
        if args.prep_fsl_randomise:
            prep_fsl_randomise(in_fnames, subrois, metrics, args)

    # use randomise (independent from prep_fsl_randomise)
    if args.use_randomise:
        for subroi,metric,atlas in itertools.product(subrois,metrics,atlases):
            use_randomise(subjs, subroi, atlas, metric, args)
            if args.plot_figs:
                plot_randomise_outputs(subjs, subroi, metric, args)

    if args.run_second_level:
        out_dir = os.path.join(baseline_dir, 'postprocessing', 'glm', pre_metric)
        os.makedirs(out_dir, exist_ok=True)
        args.out_dir = out_dir
        glm_results = run_second_level(subjs, metrics, subrois, args)

        if args.plot_within_group_masks:
            plot_within_group_masks(subrois, glm_results, args)

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

    if args.non_parametric_analysis:
        suffix = 'non_parametric'
        if args.two_sided_within_group:
            suffix += '_within2tailed'
        if args.two_sided_between_group:
            suffix += '_between2tailed'

        np_results = dict()
        for seed,metric in itertools.product(subrois, metrics):
            neg_log_pvals, mask = non_parametric_analysis(subjs, seed, metric, pre_metric, args=args, masks=[])
            plot_non_param_maps(neg_log_pvals, seed, suffix, args)
            np_results[seed] = {'neg_log_pvals':neg_log_pvals, 'mask':mask}

        if args.save_outputs:
            with gzip.open(os.path.join(baseline_dir, 'postprocessing', 'non_parametric', suffix+'.pkl.gz'), 'wb') as f:
                pickle.dump(np_results,f)

    if args.within_mask_corr:
        df_mask_corr = compute_FC_within_masks(subjs, np_results, args=args)
        if args.plot_figs:
            plot_within_mask_corr(df_mask_corr, args=args)

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

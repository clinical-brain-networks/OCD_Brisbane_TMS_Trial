# Script to visualize clinical trial fMRI analysis outputs
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
from argparse import Namespace
import bct
from datetime import datetime
import h5py
import importlib
from itkwidgets import view
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
import pandas as pd
import pdb
import pickle
import pyvista as pv
from pyvista import examples
import scipy
from scipy.io import loadmat
import seaborn as sbn
import sklearn
from sklearn.decomposition import PCA
import sys
import time
from time import time
import warnings

# paths
proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCD_clinical_trial'
code_dir = os.path.join(proj_dir, 'code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = os.path.join(proj_dir, 'utils')
fs_dir = '/usr/local/freesurfer/'

sys.path.insert(0, os.path.join(code_dir, 'functional'))
import seed_to_voxel_analysis
from seed_to_voxel_analysis import get_group, stim_coords

# uncomment in case of using freesurfer surfaces
#coords, faces, info, stamp = nib.freesurfer.io.read_geometry(os.path.join(fs_dir, 'subjects', 'fsaverage4', 'surf', 'lh.white'), read_metadata=True, read_stamp=True)

imgs_info = {   'base': {       'path': os.path.join(proj_dir, 'utils', 'empty.nii.gz'),
                                'name': 'base',
                                'cmap': 'binary',
                                'clim': [0, 1.],
                                'opacity': 0.5,
                                'nan_opacity': 0.5 },
                'stim_5mm': {   'path': os.path.join(proj_dir, 'utils', 'stim_VOI_5mm.nii.gz'),
                                'name':'stim_5mm',
                                'cmap':'Reds',
                                'clim': [0,0.5],
                                'opacity': 1,
                                'nan_opacity':0.},
                'acc_seed': {   'path': os.path.join(proj_dir, 'utils', 'Acc.nii.gz'),
                                'name': 'acc_seed',
                                'cmap': 'Reds',
                                'clim': [0,0.5],
                                'opacity': 1. ,
                                'nan_opacity':0},
                'stim_10mm': {  'path': os.path.join(proj_dir, 'utils', 'stim_VOI_10mm.nii.gz'),
                                'name':'stim_10mm',
                                'cmap':'Reds',
                                'clim': [0,0.5],
                                'opacity': 1.,
                                'nan_opacity':0.},
                'acc_pathway': {'path': os.path.join(proj_dir, 'utils', 'frontal_Acc_mapping.nii.gz'),
                                'name': 'acc_pathway',
                                'cmap': 'binary',
                                'clim': [0, 999999.],
                                'opacity': 0.8,
                                'nan_opacity':0. },
                'tian_acc': {  'path': os.path.join(proj_dir, 'utils', 'hcp_masks', 'Acc_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr100SD_GM_23092022.nii.gz'), #'Acc_100hcpThr50SD.nii.gz'), #'Acc_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr5_19092022.nii.gz'), # 'Acc_pathway_mask_group_by_hemi_Ftest_grp111_10hcpThr10SD_Fr_20092022.nii.gz'), #'Acc_fcmap_fwhm6_HCP_REST1_avg.nii'), #'nac_mask_fcmap_avg.nii.gz'),
                                'name':'tian_acc',
                                'cmap':'Oranges',
                                'clim': [0, 0.6],
                                'opacity': 1.,
                                'nan_opacity':0.},
                'group_diff': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/Acc/randomise/Acc_outputs_n5000_TFCE_group_by_session_repeated2wayANOVA_Ftest_noExBlocks_07092022_tstat1.nii.gz'),
                                'name': 'group_diff',
                                'cmap': 'RdBu',
                                'clim': [-4, 4],
                                'opacity': 1.,
                                'nan_opacity':0. },
}

group_colors = {'group1': 'orange', 'group2':'lightslategray'}

pointplot_ylim = {  'Harrison2009': {'corr': [-1,1], 'fALFF':[0.0,0.03]},
                    'TianS4': {'corr': [-0.01,0.01], 'fALFF':[0.0,0.03]},
                }

tail_colors = {'left':'firebrick', 'right':'dodgerblue'}

# Additionally, all the modules other than ipygany and pythreejs require a framebuffer, which can be setup on a headless environment with pyvista.start_xvfb().
pv.start_xvfb()

def create_stim_site_voi(stim_radius=5., args=None):
    """  create niftii image of stimuus locations of all subjects using a sphere of radius stim_radius mm """
    #xls_fname = 'MNI_coordinates_FINAL.xlsx'
    #stim_coords = pd.read_excel(os.path.join(proj_dir, 'data', xls_fname), usecols=['P ID', 'x', 'y', 'z'])
    # now global... not needed

    stim_sites = []
    for i,stim in stim_coords.iterrows():
        x,y,z = stim['x'], stim['y'], stim['z']
        stim_sites.append(nltools.create_sphere([x,y,z], radius=stim_radius))

    mean_stim_sites = mean_img(stim_sites)
    if args.save_outputs:
        nib.save(mean_stim_sites, os.path.join(proj_dir, 'utils', 'stim_VOI_'+str(stim_radius)+'mm.nii.gz'))
    return mean_stim_sites

def get_brainnet_surf(surf_name):
    """ Import brain net viewer surface into pyvista polyData type """
    brainnet_path = '/home/sebastin/Downloads/BrainNetViewer/BrainNet-Viewer/Data/SurfTemplate/'
    fname = os.path.join(brainnet_path, surf_name+'.nv')
    with open(fname, 'r') as f:
        n_vertices = int(f.readline())

    icbm_txt = pd.read_csv(fname, sep=' ', header=None, index_col=False, skiprows=[0,n_vertices+1])
    coords = np.array(icbm_txt.iloc[:n_vertices])
    faces = np.array(icbm_txt.iloc[n_vertices:], dtype=int) - 1

    nfaces, fdim = faces.shape
    c = np.ones((nfaces,1))*fdim
    icbm_surf = pv.PolyData(coords, np.hstack([c,faces]).astype(int))
    return icbm_surf, coords, faces


def volume_to_surface(vol_img, coords, faces, radius=5.):
    """ project volume niftii image to cortical surface mesh """
    left_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.left, faces.left], radius=radius, interpolation='linear')
    right_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.right, faces.right], radius=radius, interpolation='linear')
    both_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.both, faces.both], radius=radius, interpolation='linear')
    return Namespace(**{'left':left_surf, 'right':right_surf, 'both':both_surf})


def get_icbm_surf(args):
    """ imports ICBM152 surfaces into Namespace """
    if args.smoothed_surface:
        icbm_left, coords_left, faces_left = get_brainnet_surf('BrainMesh_ICBM152Left_smoothed')
        icbm_right, coords_right, faces_right = get_brainnet_surf('BrainMesh_ICBM152Right_smoothed')
        icbm_both, coords_both, faces_both = get_brainnet_surf('BrainMesh_ICBM152_smoothed')
    else:
        icbm_left, coords_left, faces_left = get_brainnet_surf('BrainMesh_ICBM152Left')
        icbm_right, coords_right, faces_right = get_brainnet_surf('BrainMesh_ICBM152Right')
        icbm_both, coords_both, faces_both = get_brainnet_surf('BrainMesh_ICBM152')
    coords = Namespace(**{'left':coords_left, 'right':coords_right, 'both':coords_both})
    faces = Namespace(**{'left':faces_left, 'right':faces_right, 'both':faces_both})
    surfs = Namespace(**{'left':icbm_left, 'right':icbm_right, 'both':icbm_both})
    return surfs, coords, faces


def project_surface(template, img, name):
    """ fill template surface with img surface data for rendering """
    img.left[img.left==0] = np.NaN
    img.right[img.right==0] = np.NaN
    img.both[img.both==0] = np.NaN
    template.left.point_data[name] = img.left
    template.right.point_data[name] = img.right
    template.both.point_data[name] = img.both


def plot_surface(surfs, stim_spheres, roi_spheres, names=imgs_info.keys(), args=None):
    """  """
    cam_pos = {'front':[-3, 2, -1], 'medial':[1,1,-0.3]}

    # Plot
    pl = pv.Plotter(window_size=[800, 600], shape=(1,1), border=False)
    pl.set_plot_theme = 'document'

    # between-group row
    """pl.subplot(0,0)
    for img_name, img_info in imgs_info.items():
        pl.add_mesh(surfs.left.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=1., interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['front']
    pl.background_color = 'white'

    pl.subplot(0,1)
    for img_name, img_info in imgs_info.items():
        pl.add_mesh(surfs.left.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=1., interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['medial']
    pl.background_color = 'white' """

    #pl.subplot(0,2)
    for img_name in names:
        img_info = imgs_info[img_name]
        pl.add_mesh(surfs.right.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=True)
        pl.add_mesh(surfs.left.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=True)
    pl.camera_position = cam_pos['front']
    pl.background_color = 'white'

    if args.show_stim_balls:
        for s in stim_spheres:
            pl.add_mesh(s['sphere'], color=s['color'])

    if args.show_roi_degree:
        for s in roi_spheres:
            pl.add_mesh(s['sphere'], color=s['color'])

    """pl.subplot(0,3)
    for img_name, img_info in imgs_info.items():
        pl.add_mesh(surfs.right.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=1., interpolate_before_map=True, show_scalar_bar=False)
    pl.camera_position = cam_pos['medial']
    pl.background_color = 'white'"""

    fname = '_'.join(names)+'_'+datetime.now().strftime('%d%m%Y')+'.pdf'
    pl.save_graphic(os.path.join(proj_dir, 'img', fname))

    pl.show(jupyter_backend='panel')
    pl.deep_clean()


def get_stim_spheres(args):
    """ create sphere of radius given in args around the stim site for each patient, return a PyVista PolyData object """
    stim_spheres = []
    for i,stim in stim_coords.iterrows():
        grp = get_group(stim['subjs'])
        if grp != 'none':
            s = pv.Sphere(center=np.array(stim[['x','y','z']], dtype=float)*args.stim_balls_scaling,
                          radius=args.stim_balls_radius)
            stim_spheres.append( {'subj':stim['subjs'], 'group':grp, 'sphere':s, 'color':group_colors[grp]} )
    return stim_spheres

def get_roi_spheres(args):
    """ create spheres of radius equal to the degree of the node connectivity """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_atlas.pkl'), 'rb') as f:
        df_atlas = pickle.load(f)
    roi_spheres = []
    for i,row in df_atlas.iterrows():
        #x,y,z = row['centroid']
        for tail in tail_colors.keys():
            degree = row['degree_'+tail]
            s = pv.Sphere(center=np.array(row['centroid'], dtype=float), radius=degree/3)
            roi_spheres.append({'sphere':s, 'color':tail_colors[tail]})
    return roi_spheres


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--surface_template', type=str, default='icbm', action='store', help='defines which surface template to use. default: icbm')
    parser.add_argument('--show_stim_balls', default=False, action='store_true', help='display small balls at stim locations')
    parser.add_argument('--stim_balls_radius', type=float, default=2, action='store', help='radius of small balls at stim location')
    parser.add_argument('--stim_balls_scaling', type=float, default=1., action='store', help='scaling of coordinates of small balls at stim location to be closer to surface mesh')
    parser.add_argument('--plot_surface', default=False, action='store_true', help='plot surface mesh with stim  locations and mask')
    parser.add_argument('--smoothed_surface', default=False, action='store_true', help='use smooth cortical mesh')
    parser.add_argument('--show_roi_degree', default=False, action='store_true', help='display spheres of degree radius at roi centroids ')
    args = parser.parse_args()

    names = ['base']

    if args.plot_surface:
        # get template ICBM surfaces (left, right and both hemispheres meshes)
        surfs, coords, faces = get_icbm_surf(args)

        # create projections on surfaces
        for img_name in names:
            img_info = imgs_info[img_name]
            # load image of interest
            img = load_img(img_info['path'])
            # project image to surface
            img_surfs = volume_to_surface(img, coords, faces)
            # prepare surface for rendering/plotting
            project_surface(surfs, img_surfs, name=img_name)

        # get small balls located at stim sites
        stim_spheres = get_stim_spheres(args)

        roi_spheres = get_roi_spheres(args)

        plot_surface(surfs, stim_spheres, roi_spheres, names=names, args=args)

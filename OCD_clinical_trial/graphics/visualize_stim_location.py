# Script to be used in Jupyter Lab to visualize TMS stimulation sites
#
# Author: Sebastien Naze
#
# QIMR Berghofer 2022

# n.b.: script to be run on Lucky3

import argparse
import bct
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
from nilearn.image import load_img, binarize_img, threshold_img, mean_img, new_img_like, math_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_img, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nltools
import numpy as np
import os
import pickle
import pandas as pd
import pyvista as pv
from pyvista import examples
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.decomposition import PCA
import sys
import time
from time import time
import warnings


proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/OCD_clinical_trial/'
ocd_baseline = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'code')
data_dir = os.path.join(proj_dir, 'data')
fs_dir = '/usr/local/freesurfer/'

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


# import ICBM152 surfaces
icbm_left, coords_left, faces_left = get_brainnet_surf('BrainMesh_ICBM152Left_smoothed')
icbm_right, coords_right, faces_right = get_brainnet_surf('BrainMesh_ICBM152Right_smoothed')
icbm_both, coords_both, faces_both = get_brainnet_surf('BrainMesh_ICBM152_smoothed')


# import stim locations
xls_fname = 'MNI_coordinates_FINAL.xlsx'
stim_coords = pd.read_excel(os.path.join(proj_dir, 'data', xls_fname), usecols=['P ID', 'x', 'y', 'z'])

stim_sites = []
for i,stim in stim_coords.iterrows():
    x,y,z = stim['x'], stim['y'], stim['z']
    stim_sites.append(nltools.create_sphere([x,y,z], radius=2))

mean_stim_sites = mean_img(stim_sites)

plot_stat_map(mean_stim_sites, threshold=0, colormap='Blues', cut_coords=[8,64,-10])

# display Acc-Fr fronto-striatal maps
acc_map_img = os.path.join(proj_dir, 'utils', 'frontal_Acc_mapping.nii.gz')
caud_map_img = os.path.join(proj_dir, 'utils', 'frontal_vPut_mapping.nii.gz')
view_img(math_img('img1 + 2*img2', img1=acc_map_img, img2=caud_map_img), cut_coords=[8,64,-10])

# display Acc-Fr fronto-striatal maps
OFC_R_img = os.path.join(ocd_baseline, 'postprocessing/SPM/seeds_and_rois/OFC_R.nii.gz')
plot_stat_map(OFC_R_img, cut_coords=[8,64,-10])

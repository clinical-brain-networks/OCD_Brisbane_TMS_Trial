#!/bin/bash
#
# Script to prepare individual subjects files to perform SPM
#

module load fsl/6.0.1
module load miniconda3/current

#source activate /mnt/lustre/working/lab_lucac/lukeH/ljh-neuro
source activate /mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/envs/hpc

proj_dir=/mnt/lustre/working/lab_lucac/sebastiN/projects/OCD_clinical_trial/
pp_dir=${proj_dir}postprocessing/

# load specific subject from subject list
mapfile -t subj_array < ${proj_dir}code/patients_list.txt
subj=sub-patient01
#IDX=$((PBS_ARRAY_INDEX-1))  # double parenthesis needed for arithmetic operations
#subj=${subj_array[$IDX]}
#echo "Current subject: " ${subj}

#mkdir -p ${pp_dir}${subj}/spm/scans/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_gsr_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_gsr_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/*

#echo 'Denoising '${subj}
#python ${proj_dir}docs/code/preprocessing/post_fmriprep_denoising.py ${subj}

echo 'Preparing '$subj' for seed-to-voxel analysis'
python ${proj_dir}code/functional/seed_to_voxel_analysis.py --subj $subj --compute_seed_corr --merge_LR_hemis --n_jobs 1


#echo 'Preparing for SPM '${subj}

#mkdir -p ${pp_dir}${subj}/spm/scans/detrend_filtered_gsr_smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_gsr_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/detrend_filtered_gsr_smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_gsr_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/detrend_filtered_gsr_smooth6mm/*

#mkdir -p ${pp_dir}${subj}/spm/scans/detrend_gsr_scrub_smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_gsr_scrub_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/detrend_gsr_scrub_smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_gsr_scrub_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/detrend_gsr_scrub_smooth6mm/*

#mkdir -p ${pp_dir}${subj}/spm/scans/smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/smooth6mm/*

#mkdir -p ${pp_dir}${subj}/spm/scans/filtered_smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-filtered_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/filtered_smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-filtered_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/filtered_smooth6mm/*

#mkdir -p ${pp_dir}${subj}/spm/scans/detrend_smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/detrend_smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/detrend_smooth6mm/*

#mkdir -p ${pp_dir}${subj}/spm/scans/detrend_filtered_smooth6mm/
#fslsplit ${proj_dir}data/derivatives/post-fmriprep-fix/${subj}/func/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_smooth-6mm.nii.gz  ${pp_dir}${subj}/spm/scans/detrend_filtered_smooth6mm/${subj}_task-rest_space-MNI152NLin2009cAsym_desc-detrend_filtered_smooth-6mm
#gzip -d ${pp_dir}${subj}/spm/scans/detrend_filtered_smooth6mm/*

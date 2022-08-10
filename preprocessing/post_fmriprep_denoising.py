"""
Wrapper code that uses Paula's 'fmripop' code to perform basic denoising.
For this project denoising is very minimal because FIX has already been run.

see https://github.com/brain-modelling-group/fmripop/blob/master/post_fmriprep.py

"""
# %%
import json
import nilearn
from nilearn.image import new_img_like
import numpy as np
import os
import platform
import sys
import time

#code_dir = '/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/docs/code/'
#sys.path.insert(0, code_dir)
#sys.path.insert(0, os.path.join(code_dir, 'preprocessing'))

# import my own code
#from functions.data_helpers import get_computer, make_dirs

# get computer name to set paths
if platform.node()=='qimr18844':  # Lucky3
    working_dir = '/home/sebastin/working/'
    computer = 'lucky3'
elif 'hpcnode' in platform.node(): # HPC
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

in_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCD_clinical_trial/')
conf_dir = in_dir+'data/derivatives/fmriprep/'
bold_dir = in_dir+'data/derivatives/fmriprep-fix/'
out_dir = os.path.join(in_dir, 'data/derivatives/post-fmriprep-fix/')

# add Paula's fmripop to path and import functions
fmripop_path = os.path.join(working_dir, 'lab_lucac/sebastiN/fmripop/')
sys.path.insert(0, fmripop_path)
from post_fmriprep import parser, fmripop_check_args, fmripop_remove_confounds, fmripop_scrub_data, fmripop_smooth_data

# define subj
subj = sys.argv[1]
print(subj)

# files and pipelines
img_space = 'MNI152NLin2009cAsym'

for ses in ['ses-pre', 'ses-post']:
    # get files for this subject
    task_nii = os.path.join(bold_dir,subj,ses,'func', subj+'_'+ses+'_task-fearRev_space-'
                        + img_space+'_desc-preproc_bold.nii.gz')
    rest_nii = os.path.join(bold_dir,subj,ses,'func', subj+'_'+ses+'_task-rest_space-'
                        + img_space+'_desc-preproc_bold.nii.gz')

    task_msk = os.path.join(bold_dir,subj,ses,'func', subj+'_'+ses+'_task-fearRev_space-'
                        + img_space+'_desc-brain_mask.nii.gz')
    rest_msk = os.path.join(bold_dir,subj,ses,'func', subj+'_'+ses+'_task-rest_space-'
                        + img_space+'_desc-brain_mask.nii.gz')

    task_tsv = os.path.join(conf_dir,subj,ses,'func', subj+'_'+ses+'_task-fearRev_desc-confounds_timeseries.tsv')
    rest_tsv = os.path.join(conf_dir,subj,ses,'func', subj+'_'+ses+'_task-rest_desc-confounds_timeseries.tsv')

    # list the models I would like to run:

    pipelines = {'detrend_gsr_filtered_scrubFD05': { 'niipath': rest_nii,
                                                  'maskpath': rest_msk,
                                                  'tsvpath': rest_tsv,
                                                  'add_orig_mean_img': True,
                                                  'confound_list': ['global_signal'],
                                                  'detrend': True,
                                                  'fmw_disp_th': 0.5,
                                                  'fwhm': 0,
                                                  'high_pass': 0.01,
                                                  'low_pass': 0.1,
                                                  'scrubbing': True,
                                                  'remove_volumes': True,
                                                  'tr': 0.81,
                                                  'num_confounds': 1,
                                                  'task': 'rest'},
                'detrend_gsr_smooth-6mm': { 'niipath': rest_nii,
                                                  'maskpath': rest_msk,
                                                  'tsvpath': rest_tsv,
                                                  'add_orig_mean_img': True,
                                                  'confound_list': ['global_signal'],
                                                  'detrend': True,
                                                  'fmw_disp_th': None,
                                                  'fwhm': 6,
                                                  'high_pass': None,
                                                  'low_pass': None,
                                                  'scrubbing': False,
                                                  'remove_volumes': False,
                                                  'tr': 0.81,
                                                  'num_confounds': 1,
                                                  'task': 'rest'}}


    for pl_label in pipelines:
        print('Running: '+pl_label)
        # use my own wrapper code (similar to __main__ in fmripop)
        start_time = time.time()
        pl = pipelines[pl_label]

        # set up args obj
        args = parser.parse_args('')

        # Modify the arguments based on dict
        args.niipath = pl['niipath']
        args.maskpath = pl['maskpath']
        args.tsvpath = pl['tsvpath']
        args.add_orig_mean_img = pl['add_orig_mean_img']
        args.confound_list = pl['confound_list']
        args.detrend = pl['detrend']
        args.fmw_disp_th = pl['fmw_disp_th']
        args.fwhm = pl['fwhm']
        args.high_pass = pl['high_pass']
        args.low_pass = pl['low_pass']
        args.num_confounds = pl['num_confounds']
        args.remove_volumes = pl['remove_volumes']
        args.scrubbing = pl['scrubbing']
        args.tr = pl['tr']

        # Set derived Parameters according to user specified parameters
        args = fmripop_check_args(args)

        # Convert to dict() for saving later
        params_dict = vars(args)
        params_dict['fwhm'] = args.fwhm.tolist()

        # Performs main task -- removing confounds
        out_img = fmripop_remove_confounds(args)

        # Perform additional actions on data
        if args.scrubbing:
            out_img, params_dict = fmripop_scrub_data(out_img, args, params_dict)

        if np.array(args.fwhm).sum() > 0.0:  # If fwhm is not zero, performs smoothing
            out_img = fmripop_smooth_data(out_img, args.fwhm)

        # Save output image and parameters used in this script
        out_path = os.path.join(out_dir,subj,ses,'func/')
        out_file = (out_path+subj+'_'+ses+'_task-'+pl['task']+'_space-'
                    + img_space+'_desc-'+pl_label+'.nii.gz')
        os.makedirs(out_path, exist_ok=True)

        # make sure the out img has the correct header
        out_img = new_img_like(
            pl['niipath'], out_img.get_fdata(), copy_header=True)

        # Save the clean data in a separate file
        out_img.to_filename(out_file)

        # Save the input arguments in a json file with a timestamp
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        out_file = 'fmripop_'+pl_label+'_parameters.json'
        with open(os.path.sep.join((out_path, out_file)), 'w') as file:
            file.write(json.dumps(params_dict, indent=4, sort_keys=True))

        print("--- %s seconds ---" % (time.time() - start_time))

print('Finished all pipelines')

# %%

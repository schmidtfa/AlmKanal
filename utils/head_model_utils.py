from os.path import join
import numpy as np
import mne
from mne.coreg import Coregistration
from os.path import join
import os
import pickle

def compute_headmodel(raw,
                      subject_id,
                      base_data_path,
                      savefig=True):
    
    '''
    base_data_path: The path to the data directory where freesurfer and the headmodels are going to be stored.

    '''
    
    mri_path = join(base_data_path, 'freesurfer')
    out_folder = join(base_data_path, 'headmodels', subject_id)
    trans = 'fsaverage' 
    
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=True))

    #%% do the coregistration
    coreg = Coregistration(info, trans, mri_path)
    coreg.set_scale_mode('3-axis')
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
    coreg.omit_head_shape_points(distance=5 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "f"/ {np.min(dists):.2f} mm / "
        f"{np.max(dists):.2f} mm")
    
    #%% create and save a scaled copy of mri subject fs average
    new_source_identifier = subject_id + '_from_template'

    #Dont forget this sdcales bem, atlas and source space automatically too
    mne.coreg.scale_mri("fsaverage", new_source_identifier, 
                        scale=coreg.scale, 
                        subjects_dir=mri_path, 
                        annot=True, overwrite=True)

    if os.path.isdir(out_folder) == False:
        os.makedirs(out_folder)
    file = open(join(out_folder, subject_id) + "info.pickle", 'wb')
    pickle.dump(info, file)
    file.close()    
    
    mne.write_trans(join(out_folder, subject_id) + '-trans.fif', coreg.trans, overwrite=True)
    print('Coregistration done!')

    if savefig: # REDO IT LATER
            
        # PLACEHOLDER: plot 3d stuff when xvfb is available
        import matplotlib.pyplot as plt  # Assuming 'info' is your data structure containing sensor and digitization information
        head_mri_t = mne.transforms._get_trans(coreg.trans, "head", "mri")[0]
        coord_frame = "head"
        to_cf_t = mne.transforms._get_transforms_to_coord_frame(info, head_mri_t, coord_frame=coord_frame)

        sensor_locs = np.array([ch['loc'][:3] for ch in info['chs'] if ch['ch_name'].startswith('MEG')])
        sensor_locs = mne.transforms.apply_trans(to_cf_t['meg'], sensor_locs)

        # Extract Digitization Points (excluding fiducials)
        head_shape_points = np.array([point['r'] for point in info['dig'] if point['kind'] == 4])
        head_shape_points = mne.transforms.apply_trans(to_cf_t['head'], head_shape_points)

        # Create a 2x2 subplot layout
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f'MEG - DIG Coregistration of {subject_id}', fontsize=16)

        # Axial view
        ax1 = fig.add_subplot(221)
        ax1.scatter(sensor_locs[:, 0], sensor_locs[:, 1], s=20, c='r', label='Sensors')
        ax1.scatter(head_shape_points[:, 0], head_shape_points[:, 1], s=10, c='b', label='Head Shape')
        ax1.set_title('Axial View')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Distance (m)')
        ax1.legend()

        # Coronal view
        ax2 = fig.add_subplot(222)
        ax2.scatter(sensor_locs[:, 0], sensor_locs[:, 2], s=20, c='r')
        ax2.scatter(head_shape_points[:, 0], head_shape_points[:, 2], s=10, c='b')
        ax2.set_title('Coronal View')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Distance (m)')

        # Sagittal view
        ax3 = fig.add_subplot(223)
        ax3.scatter(sensor_locs[:, 1], sensor_locs[:, 2], s=20, c='r')
        ax3.scatter(head_shape_points[:, 1], head_shape_points[:, 2], s=10, c='b')
        ax3.set_title('Sagittal View')
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Distance (m)')

        # 3D plot
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(sensor_locs[:, 0], sensor_locs[:, 1], sensor_locs[:, 2], s=20, c='r', label='Sensors')
        ax4.plot(sensor_locs[:, 0], sensor_locs[:, 1], sensor_locs[:, 2], color='k', linewidth=0.5)  # Connect sensors with lines
        ax4.scatter(head_shape_points[:, 0], head_shape_points[:, 1], head_shape_points[:, 2], s=10, c='b', label='Head Shape')
        ax4.set_title('3D View')
        ax4.grid(False)  # Remove grid
        ax4.axis('off')  # Remove axis

        plt.tight_layout()
        # save 
        plt.savefig(join(out_folder, subject_id) +  '_coreg.png', dpi=300)
        # not sohw ... plt.show()



def make_fwd(info, source, trans_path, subjects_dir, subject_id, template_mri):
    
    ###### MAKE FORWARD SOLUTION AND INVERSE OPERATOR
    if template_mri:
        fpath_add_on = '_from_template'
    else:
        fpath_add_on = ''

    fs_path = join(subjects_dir, f'{subject_id}{fpath_add_on}')
    bem_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-5120-5120-5120-bem.fif'

    if source == 'volume':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-vol-10-src.fif'
        
    elif source == 'surface':
        src_file = f'{fs_path}/bem/{subject_id}{fpath_add_on}-ico-4-src.fif'
    
    fname_trans = join(trans_path, subject_id, subject_id + '-trans.fif')

    bem_sol = mne.make_bem_solution(bem_file, 
                                    solver='mne', 
                                    verbose=True) 
    fwd = mne.make_forward_solution(info=info, 
                                    trans=fname_trans, 
                                    src=src_file, 
                                    bem=bem_sol)

    return fwd

import mne
import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def plot_parc(
    stc_parc: NDArray,
    stc_mask: NDArray,
    labels_mne: list,
    subjects_dir: str,
    cmap: str,
    clevels: list,
    plot_kwargs: dict,
    parc: str = 'HCPMMP1',
) -> NDArray:
    # mpl.use('Qt5Agg')

    labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

    names_order_mne = np.array([label.name[:-3] for label in labels_mne])

    rh = [label.hemi == 'rh' for label in labels_mne]
    lh = [label.hemi == 'lh' for label in labels_mne]

    brain = mne.viz.get_brain_class()  # doesnt work directly from pysurfer

    brain = brain('fsaverage', **plot_kwargs)

    # mask locations based on percentile
    for hemi in ['lh', 'rh']:
        annot_file = subjects_dir + f'/fsaverage/label/{hemi}.{parc}.annot'
        labels, _, nib_names = nib.freesurfer.read_annot(annot_file)

        names_order_nib = np.array([str(name)[2:-1] for name in nib_names])

        if hemi == 'lh':
            names_mne = names_order_mne[lh]
            cur_stc = stc_parc[lh]  # , tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[lh]
        else:
            names_mne = names_order_mne[rh]
            cur_stc = stc_parc[rh]  # , tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[rh]

        # Create a dictionary to map strings to their indices in array1
        index_dict = {value: index for index, value in enumerate(names_mne)}

        # Find the indices of strings in array1 corresponding to array2
        right_order = [index_dict[value] for value in names_order_nib]

        cur_stc_ordered = cur_stc[right_order]
        cur_mask_ordered = cur_mask[right_order]

        cur_stc_ordered[cur_mask_ordered] = np.nan

        vtx_data = cur_stc_ordered[labels]
        vtx_data[labels == -1] = -1

        brain.add_data(
            vtx_data,
            hemi=hemi,
            fmin=clevels[0],
            fmid=clevels[1],
            fmax=clevels[2],
            colormap=cmap,  # np.nanmax(stc_parc)
            colorbar=False,
            alpha=0.8,
        )

    screenshot = brain.screenshot()
    # brain.close()

    return screenshot

from compute_models import *

excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

def trunc_motion_samples(selected, excluded):
    """Truncate motion samples
    :return truncated motion samples in a dictionary by filename"""
    included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
    data = parse_selected(selected, limit=None)
    trunc_motion_samples = {}
    for key in data.keys():
        for i, fname in enumerate(data[key]['actionid']):
            sample_data = {key:{}}
            sample_data[key]['actions'] = [data[key]['actions'][i]]
            sample_data[key]['labels'] = [data[key]['labels'][i]]
            X, y = gather_all_np(sample_data)
            X = X[:, :(X.shape[1] - 3)]
            X = remove_excluded(truncate(X),indices, type='numpy')
            trunc_motion_sample = array2motions(X, indices, type='numpy')
            trunc_motion_samples[fname] = trunc_motion_sample
    return trunc_motion_samples

# lav selected dictionary her. Fx.
selected = get_manual_names(["walk"])

truncated_motions = trunc_motion_samples(selected, excluded)



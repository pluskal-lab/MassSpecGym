import numpy as np
from pprint import pprint

def fix_jss(x):

    y = 1. - (1. - x / 100.) / np.log(2.)
    return y


metrics_d = {
    "retform_fp": (fix_jss(47.52), fix_jss(46.89), fix_jss(48.18)),
    "retform_gnn": (fix_jss(44.26), fix_jss(43.70), fix_jss(44.87)),
    "retform_preconly": (fix_jss(40.85), fix_jss(40.26), fix_jss(41.49)),
    "retform_fragnnet": (fix_jss(63.28), fix_jss(62.75), fix_jss(63.81)),
    # "retmass_fp": (fix_jss(47.80), fix_jss(47.17), fix_jss(48.47)),
    # "retmass_gnn": (fix_jss(44.15), fix_jss(43.56), fix_jss(44.75)),
    # "retmass_preconly": (fix_jss(40.85), fix_jss(40.27), fix_jss(41.46)),
}

for k, v in metrics_d.items():
    print(f"{k}: {np.around(v[0],decimals=2):.2f} ({np.around(v[1],decimals=2):.2f},{np.around(v[2],decimals=2):.2f})")

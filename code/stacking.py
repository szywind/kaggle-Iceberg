# References: https://www.kaggle.com/dongxu027/explore-stacking-lb-0-1463

import os
import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


sub_path = "../submit/stacking/0112"
all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = len(concat_sub.columns)


# check correlation
concat_sub.corr()


# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)



# set up cutoff threshold for lower and upper bounds, easy to twist
cutoff_lo = 0.8
cutoff_hi = 0.2


concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1),
                                    concat_sub['is_iceberg_max'],
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'],
                                             concat_sub['is_iceberg_median']))
concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median_{}.csv'.format(ncol-1),
                                        index=False, float_format='%.6f')

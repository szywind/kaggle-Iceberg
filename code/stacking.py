# References: https://www.kaggle.com/dongxu027/explore-stacking-lb-0-1463

import os
import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


def stacking_trainable(feat_path = "../features", submit_path = "../submit/stacking/0116"):
    all_files = os.listdir(feat_path)
    outs = {f[f.rfind('ensemble')+len('ensemble')+1: f.rfind('.csv')]: pd.read_csv(os.path.join(feat_path, f), index_col=0) for f in all_files}
    ncol = len(outs)
    nrow = len(pd.read_csv(os.path.join(feat_path, all_files[0]), index_col=0))
    X = np.ones((nrow, ncol), dtype=np.float32)
    y = np.ones((nrow, 1), dtype=np.float32)

    # models = []
    models = list(outs.keys())

    for i, j in enumerate(outs):
        X[:, i] = outs[j].iloc[:,0]
        y = outs[j].iloc[:,1]
        # models.append(j)

    from sklearn import linear_model
    ridgereg = linear_model.RidgeCV(cv=5)
    ridgereg.fit(X=X, y=y)
    logreg = linear_model.LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear')
    logreg.fit(X=X, y=y)

    # prediction
    test_files = os.listdir(submit_path)
    test_outs = {f[f.rfind('ensemble')+len('ensemble')+1: f.rfind('.csv')]: pd.read_csv(os.path.join(submit_path, f), index_col=0) for f in test_files}
    test_df = pd.read_csv(os.path.join(submit_path, test_files[0]))
    num_test = len(test_df)
    X_test = np.ones((num_test, ncol))
    for i, m in enumerate(models):
        X_test[:,i] = test_outs[m].iloc[:,0]

    ## check convergence of the logistic regression
    # y_pred = logreg.predict(X)
    # print(np.sum(y == y_pred) / len(y))

    # y_pred = logreg.predict_proba(X)
    # print(y_pred.shape)
    # print(np.sum((y_pred[:,1] > 0.5) == y) / len(y))

    # y_pred = ridgereg.predict(X)
    # print(np.sum((y_pred > 0.5) == y) / len(y))

    ## logistic regression
    y_pred = logreg.predict_proba(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    # pred_df = test_df[['id']].copy()
    test_df['is_iceberg'] = y_pred[:, 1]
    test_df.to_csv('stacking_{}.csv'.format('logreg'), index=False)

    ## ridge regression
    y_pred = ridgereg.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    test_df['is_iceberg'] = y_pred
    test_df.to_csv('stacking_{}.csv'.format('ridgereg'), index=False)

def stacking_non_trainable(sub_path = "../submit/stacking/0116"):
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

if __name__ == "__main__":
    stacking_non_trainable()
    stacking_trainable()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:20:01 2019

@author: smouz

"""

import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report, auc, log_loss, accuracy_score, f1_score, recall_score, roc_curve, make_scorer

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures, PowerTransformer

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,\
 AdaBoostClassifier, BaggingClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier

from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

print('Current working dir:', os.getcwd())
#%%
# =============================================================================
# IMPORT DATA
# =============================================================================
# test set file
test_file = os.path.join('..', 'data', 'find-the-fund', 'test.csv')
fund_test_df = pd.read_csv(test_file, parse_dates=True, low_memory=False)
fund_test_df.info()

# train set file
train_file = os.path.join('..', 'data', 'find-the-fund', 'train.csv')
fund_train_df = pd.read_csv(train_file, parse_dates=True, low_memory=False)
fund_train_df.info()

# sample submision file
submit_file = os.path.join('..', 'data', 'find-the-fund', 'sampleSubmission.csv')
sample_submit = pd.read_csv(submit_file)
sample_submit.head()



#%%
def percent_missing(df, threshold=50, drop=False):
    """
    Calculate percent missing values in each column.

    Returns a series showing the proportion of missing values in each column

    Drop columns which contain more than <threshold> missing values.

    Returns:
    --------
    pd.DataFrame

    """
    ms = (np.sum(df.isnull()) / len(df)) * 100
    if drop:
        return df.drop(labels=ms[ms >= threshold].index, axis=1)
    else:
        return ms.sort_values()

# select numeric dtype
def get_numeric_data(df):
    return df.select_dtypes(include=[np.float64, np.int64])

def fix_row(error_value, df, col_name):
    df.loc[df[col_name] == error_value, col_name] = np.nan

def to_frequency(series, df):
    freq = (df[series].dropna().value_counts() / len(df[series].dropna()))
    return df[series].map(freq)


def preprocess_df(fd, train=False):
    fd = percent_missing(fd, 26, drop=True)

    # LOWERCASE
    # extract 'object' columns
    obj_cols = [col for col in fd.columns.values if fd[col].dtype == np.object]

    # chain methods to convert to lowercase
    fd[obj_cols] = fd[obj_cols].applymap(str).applymap(str.lower)



    # CONVERT DTYPE
    to_float_cols = ['total_funding_usd', 'funding_rounds', 'num_investors']
    # handle error
    for column in to_float_cols:
        while fd[column].dtypes != 'float':
            err = ''
            try:
                fd[column] = fd[column].astype('float')
            except ValueError as ve:
                # extract error using regular exp
                err = ve
                value_err = re.findall(r"\'.*", err.args[0])[0].strip("\''")
                fix_row(value_err, fd, column)
                print(err)

    # copy df
    fd_clean = fd.copy()

    if train:
#        q95 = np.quantile(fd_clean['total_funding_usd'].dropna(), [0.95])
#        fd_clean = fd_clean[(fd_clean['total_funding_usd'] < q95[0])]
##       PROCESS OUTLIERS
#        q1, q3 = np.percentile(fd_clean['total_funding_usd'].dropna(), [25, 75])
#        iqr = q3 - q1
#        upper = q3 + (iqr * 1.5)
#        lower = q1 - (iqr * 1.5)
#        fd_clean = fd_clean[(fd_clean['total_funding_usd'] >= lower) &\
#                            (fd_clean['total_funding_usd'] <= upper)]



        bool_mask = ((fd_clean['successful_investment'] != 0) & \
                     (fd_clean['successful_investment'] != 1))
        fd_clean.loc[bool_mask, ['successful_investment']] = 1


    # CREATE NEW FEATURES
    # convert categorical to frequency
    fd_clean['domain_freq'] = to_frequency('domain', fd_clean)**(1/2)
    fd_clean['country_freq'] = np.log(1/((to_frequency('hq_country_code', fd_clean))))
    fd_clean['op_status_freq'] = np.log(1/(to_frequency('op_status', fd_clean)))
    fd_clean['comp_name_freq'] = (to_frequency('comp_name', fd_clean))
    #
#    fd_clean['total_funding_usd_log'] = np.log(fd_clean['total_funding_usd'].values)

    fd_clean['X1'] = fd_clean['total_funding_usd'].values * fd_clean['domain_freq'].values
    fd_clean['X2'] = fd_clean['total_funding_usd'].values * fd_clean['country_freq'].values


    return fd_clean
#%%
fd_clean_test = preprocess_df(fund_test_df)
fd_clean_train = preprocess_df(fund_train_df, True)


#success_grp = fd_clean_train.groupby('successful_investment')
#
#success_grp[['funding_duration']].mean()
#success_grp[['op_status']].count()
#
#
#succes_usd_mean = success_grp[['total_funding_usd']].median().iloc[1, :].values[0]
#fail_usd_mean = success_grp[['total_funding_usd']].median().iloc[0, :].values[0]
#
#
#fd_clean_train['success_funding_usd'] = fd_clean_train['total_funding_usd'] / succes_usd_mean
#fd_clean_train['fail_funding_usd'] = fd_clean_train['total_funding_usd'] / fail_usd_mean


#%%

def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=11, shuffle=True)
    train_feature = pd.Series(index=train.index)
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values

def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
    # Get test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    # Get train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
    # Return new features to add to the model
    return train_feature, test_feature

#%%

fd_clean_train['op_status_enc'], fd_clean_test['op_status_enc'] = mean_target_encoding(
        train=fd_clean_train,
        test=fd_clean_test,
        target='successful_investment',
        categorical='op_status',
        )

get_numeric_data(fd_clean_train).columns.to_list()

#%%

fail_subset = fd_clean_train[fd_clean_train['successful_investment'] == 0]
fail_subset['op_status'].unique()
fail_subset['op_status'].value_counts()


fd_clean_train['op_status'].unique()
fd_clean_train['op_status'].value_counts()

fd_clean_test['op_status'].unique()
fd_clean_test['op_status'].value_counts()

win_subset = fd_clean_train[fd_clean_train['successful_investment'] == 1]
win_subset['op_status'].unique()
win_subset['op_status'].value_counts()




#%%

#fd_clean_test = preprocess_df(fund_test_df)
#fd_clean_train = preprocess_df(fund_train_df, True)
#
#fd_clean_train['successful_investment'].value_counts()


features = [
#        'X1',
#        'X2',
#        'op_status_freq',
        'op_status_enc',
#        'domain_freq',
        'country_freq',
        'total_funding_usd',
#        'funding_rounds',
#        'funding_duration',
#        'num_investors',
#        'comp_name_freq',
#        'success_funding_usd',
#        'fail_funding_usd',

        ]

X = get_numeric_data(fd_clean_train)[features]
y = fd_clean_train['successful_investment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=11
                                                    )

_, X_holdout, _, y_holdout = train_test_split(X_train, y_train, test_size=0.2,
                                              stratify=y_train,
                                              shuffle=True,
                                              random_state=11
                                              )

print('X_train size:  ', X_train.shape[0])
print('X_test size:   ', X_test.shape[0])
print('X_holdout size:', X_holdout.shape[0])

X_holdout.shape[0] + X_test.shape[0] + X_train.shape[0]

y_test.value_counts()

#%%
# UPSAMPLE
# resample data to account for imbalanced data
#X_c = pd.concat([X_train, y_train], axis=1)
#
#fail = X_c[X_c['successful_investment'] == 0]
#success = X_c[X_c['successful_investment'] == 1]
#
## upsample to match 'fail' class
#success_upsampled = resample(success,
#                             replace=True,
#                             n_samples=int(len(fail)*(0.25)),
#                             random_state=11
#                             )
#
#upsample = pd.concat([fail, success_upsampled])
#
#upsample['successful_investment'].value_counts()
#
#upsample.columns.to_list()
#
## split back into X_train, y_train
#X_train = upsample.drop('successful_investment', axis=1)
#y_train = upsample['successful_investment']
#
#print('X_train size:  ', X_train.shape[0])
#print('X_test size:   ', X_test.shape[0])
#print('X_holdout size:', X_holdout.shape[0])

#%%
# DOWNSAMPLE
#X_c = pd.concat([X_train, y_train], axis=1)
#
#fail = X_c[X_c['successful_investment'] == 0]
#success = X_c[X_c['successful_investment'] == 1]
#
## downsample majority
#fail_downsampled = resample(fail,
#                                replace = False,
#                                n_samples = len(success),
#                                random_state = 11
#                                )
#
## combine minority and downsampled majority
#downsampled = pd.concat([fail_downsampled, success])
#
#downsampled['successful_investment'].value_counts()
## split back into X_train, y_train
#X_train = downsampled.drop('successful_investment', axis=1)
#y_train = downsampled['successful_investment']

#%%

# PIPELINE
def build_pipeline(clf, impute_strat='mean', **kwargs):
    # main pipeline steps
    steps = [
        ('imputer', SimpleImputer(strategy=impute_strat)),
#        ('transformer', QuantileTransformer(output_distribution='normal')),
        ('clf', clf(**kwargs))
    ]
    return Pipeline(steps)

def display_metrics(y_test, y_pred, y_train, y_train_pred):
    print('='*50)
    print('Train accuracy:', accuracy_score(y_train, y_train_pred))
    print('Train recall:  ', recall_score(y_train, y_train_pred))
    print('Train f1:      ', f1_score(y_train, y_train_pred))

    print('Test accuracy: ', accuracy_score(y_test, y_pred))
    print('Test recall:   ', recall_score(y_test, y_pred))
    print('Test f1:       ', f1_score(y_test, y_pred))
    print('='*50)

# Train model and show metrics
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    display_metrics(y_test, y_pred, y_train, y_train_pred)

#    print('='*50)
#    print('Train set accuracy: ', accuracy_score(y_train, y_train_pred))
#    print('Test set accuracy:  ', accuracy_score(y_test, y_pred))
#    print('Train set recall: ', recall_score(y_train, y_train_pred))
#    print('Test set recall:  ', recall_score(y_test, y_pred))
#    print('='*50)



# Function: Grid Search
def grid_search(model, params, cv=10, X_train=X_train, y_train=y_train, scoring_func=f1_score):
    model_search = GridSearchCV(
        model,
        param_grid=params,
        scoring=make_scorer(scoring_func),
        n_jobs=-1,
        cv=cv
    )
    model_search.fit(X_train, y_train)
    return model_search

#%%

def model_evaluation(classifier, search_params, scoring_func=f1_score,
                     quick_run=False, show_importance=False, **kwargs):

    if quick_run:
        rf = build_pipeline(
            classifier,
            **kwargs
        )
        print('\nFirst Model Results:')
        return fit_model(rf, X_train, y_train)

    rf = build_pipeline(
        classifier,
        **kwargs
    )
    print('\nFirst Model Results:')
    fit_model(rf, X_train, y_train)

    # FEATURE IMPORTANCE
    if show_importance:
        feats = pd.Series(
            rf.named_steps['clf'].feature_importances_,
            X_train.columns.to_list(),
        ).sort_values(ascending=False)
        print(feats)

        plt.figure(figsize=(12,6))
        sns.barplot(x=feats.index.to_list(),
                    y=feats.values,
                    )
        plt.title('Feature Importance')
        plt.xticks(rotation=30)
        plt.show()
        return

    # PERFORM GRIDSEARCH AND VALIDATE SCORE
    print('\nSearching for best params...')
    rf_search = grid_search(rf, search_params)

    # BEST PARAMS
    # predict using best parameters
    print('='*50)
    print('Best params:', rf_search.best_params_)
    y_pred = rf_search.predict(X_test)
    y_holdout_pred = rf_search.predict(X_holdout)

    print('='*50)
    print('Test accuracy:   ', accuracy_score(y_test, y_pred))
    print('Test recall:     ', recall_score(y_test, y_pred))
    print('Test f1:         ', f1_score(y_test, y_pred))
    print('Test log-loss:   ', log_loss(y_test, y_pred))

    print('Holdout accuracy:', accuracy_score(y_holdout, y_holdout_pred))
    print('Holdout recall:  ', recall_score(y_holdout, y_holdout_pred))
    print('Holdout f1:      ', f1_score(y_holdout, y_holdout_pred))
    print('Holdout log-loss:', log_loss(y_holdout, y_holdout_pred))
    xx,yy, _ = roc_curve(y_test, y_pred)
    print('AUC:', auc(xx,yy))
    print('='*50)

    print('***')
    print('Positive preds:')
    print(pd.Series(y_holdout_pred).value_counts())
    print(pd.Series(y_pred).value_counts())
    print('***')

    # CROSS VALIDATE MODEL
    print(f'\nValidating score...\nUsing {str(scoring_func.__name__)} metric...')
    rf_cv = cross_val_score(
        rf_search.best_estimator_,
        X_train,
        y_train,
        scoring=make_scorer(scoring_func, greater_is_better=False),
        n_jobs=-1,
        cv=10,
    )
    print('CV Mean score    :', rf_cv.mean())
    print('CV Standard dev. :', rf_cv.std())
    print('***')
    return rf_search

#%%

#                               RANDOM FOREST
# =============================================================================
rf_search_params = {
#    'clf__max_depth': np.arange(1,5),
    'clf__n_estimators': np.arange(4,12),
#    'clf__min_samples_leaf': np.arange(3,6),
#    'clf__min_samples_split': np.arange(3,6),
}

rf_model = model_evaluation(RandomForestClassifier,
                 n_estimators=10,
#                 min_samples_split=3,
#                 min_samples_leaf=5,
                 search_params=rf_search_params,
                 n_jobs=-1,
                 random_state=11,
#                 show_importance=True,
#                 quick_run=True,
                 scoring_func=f1_score,
                 class_weight='balanced'
                 )

#%%

#                               GRADIENT BOOSTING
# =============================================================================
gb_search_params = {
#    'clf__max_depth': np.arange(1,5),
    'clf__n_estimators': np.arange(10,50,10),
#    'clf__min_samples_leaf': np.arange(3,6),
    'clf__subsample': [0.8,0.9],
}

gb_model = model_evaluation(
        GradientBoostingClassifier,
        n_estimators=50,
        search_params=gb_search_params,
#        n_jobs=-1,
        random_state=11,
#        show_importance=True,
#       quick_run=True,
        scoring_func=f1_score
        )


#%%

#                                   BAGGING
# =============================================================================
bag_search_params = {
#    'clf__max_depth': np.arange(1,5),
    'clf__n_estimators': np.arange(2,6),
#    'clf__min_samples_leaf': np.arange(3,6),
    'clf__max_samples': [0.5,0.6,0.7],
}

bag_model = model_evaluation(
        BaggingClassifier,
        n_estimators=50,
        search_params=bag_search_params,
        n_jobs=-1,
        random_state=11,
#        show_importance=True,
#       quick_run=True,
        scoring_func=f1_score
        )


#%%

#                                  TREES
# =============================================================================
rf_search_params = {
    'clf__max_depth': np.arange(1,10),
#    'clf__n_estimators': np.arange(15,55,10),
    'clf__min_samples_leaf': np.arange(2,6),
#    'clf__min_samples_split': np.arange(5,10),
}

trees_model = model_evaluation(DecisionTreeClassifier,
                 max_depth=18,
#                 min_samples_split=2,
#                 min_samples_leaf=2,
                 search_params=rf_search_params,
#                 n_jobs=-1,
                 random_state=11,
#                 show_importance=True,
#                 quick_run=True,
                 scoring_func=recall_score
                 )


#%%

print('*** Logistic Regression ***')

# PIPELINE
# modify pipeline
def linear_pipeline(clf, poly_degree=0, impute_strat='mean', **kwargs):

    # main pipeline steps
    steps = [
        ('imputer', SimpleImputer(strategy=impute_strat)),
#        ('transformer', QuantileTransformer(output_distribution='normal')),
#        ('transformer', PowerTransformer()),
        ('norm', StandardScaler()),
        ('clf', clf(**kwargs))
    ]
    if poly_degree > 0:
        steps.insert(2, ('poly_features', PolynomialFeatures(degree=poly_degree)))
    return Pipeline(steps)
#%%
# =============================================================================
#                               LogisticRegression
# =============================================================================

log_reg = linear_pipeline(
    LogisticRegression,
    penalty='l1',
    C=1.05,
    class_weight='balanced',
#    poly_degree=2,
#    warm_start=True,
    n_jobs=-1,
    random_state=11,
)

fit_model(log_reg, X_train, y_train)

#%%
# GridSearch
# =============================================================================
lgr_params = {
    'clf__penalty': ['l1'],
    'clf__C': np.linspace(1.01, 1.1, 6),
#    'clf__solver': ['liblinear', 'sag', 'saga'],
}
print('\nSearching for best params...')
lgr_search = grid_search(log_reg, lgr_params)

print('='*50)
print('Best params:', lgr_search.best_params_)
y_pred = lgr_search.predict(X_test)
y_holdout_pred = lgr_search.predict(X_holdout)
print('Test score:    ', accuracy_score(y_test, y_pred))
print('Test f1:       ', f1_score(y_test, y_pred))
print('Test recall:   ', recall_score(y_test, y_pred))
print('Holdout acc:   ', accuracy_score(y_holdout, y_holdout_pred))
print('Holdout f1:    ', f1_score(y_holdout, y_holdout_pred))
print('Holdout recall:', recall_score(y_holdout, y_holdout_pred))
xx,yy, _ = roc_curve(y_test, y_pred)
print('AUC:', auc(xx,yy))
print('='*50)
print('='*50)
print('***')
print('Positive preds:')
print(pd.Series(y_holdout_pred).value_counts())
print(pd.Series(y_pred).value_counts())
print('***')

print(classification_report(y_test, y_pred))

# Cross Validate LogReg model
# =============================================================================
print('\nValidating score...')
lg_cv = cross_val_score(
        lgr_search.best_estimator_,
        X_train,
        y_train,
        scoring=make_scorer(recall_score),
        n_jobs=-1,
        cv=10,
        )

print('CV Mean recall  :', lg_cv.mean())
print('CV Standard dev.:', lg_cv.std())

#%%




# =============================================================================
#                     Linear Classifier Model Evaluation
# =============================================================================

def linear_model_evaluation(classifier, search_params, scoring_func=f1_score,
                quick_run=False, **kwargs):
    """
    Returns:
    --------
       The resulting GridSearchCV object with best parameters.
       This can be used to make final predictions on test set.

    """
    # DEFAULT PARAMS CLF
    if quick_run:
        pipe = linear_pipeline(
            classifier,
            **kwargs
        )
        return fit_model(pipe, X_train, y_train)

    pipe = linear_pipeline(
        classifier,
        **kwargs
    )
    fit_model(pipe, X_train, y_train)

    # GRIDSEARCH
    print('\nSearching for best params...')
    pipe_search = grid_search(pipe, search_params)
    print('='*50)
    print('Best params:', pipe_search.best_params_)
    y_pred = pipe_search.predict(X_test)
    y_holdout_pred = pipe_search.predict(X_holdout)
    print('='*50)
    print('Test accuracy:   ', accuracy_score(y_test, y_pred))
    print('Test recall:     ', recall_score(y_test, y_pred))
    print('Test f1:         ', f1_score(y_test, y_pred))
    print('Test log-loss:   ', log_loss(y_test, y_pred))

    print('Holdout accuracy:', accuracy_score(y_holdout, y_holdout_pred))
    print('Holdout recall:  ', recall_score(y_holdout, y_holdout_pred))
    print('Holdout f1:      ', f1_score(y_holdout, y_holdout_pred))
    print('Holdout log-loss:', log_loss(y_holdout, y_holdout_pred))
    xx,yy, _ = roc_curve(y_test, y_pred)
    print('AUC:', auc(xx,yy))
    print('='*50)
    print('='*50)

    print('Positive preds:')
    print(pd.Series(y_holdout_pred).value_counts())
    print(pd.Series(y_pred).value_counts())
    print('='*50)


    # CROSS VALIDATE MODEL
    print(f'\nValidating score...\nUsing {str(scoring_func.__name__)} metric...')
    ln_cv = cross_val_score(
        pipe_search.best_estimator_,
        X_train,
        y_train,
        scoring=make_scorer(scoring_func),
        n_jobs=-1,
        cv=10,
    )

    print('CV Mean score    :', ln_cv.mean())
    print('CV Standard dev. :', ln_cv.std())
    print('***')
    return pipe_search
#%%
# =============================================================================
#                               SGDClassifier
# =============================================================================
sgd_search = {
        'clf__penalty': ['l1'],
        'clf__epsilon': [0.0001,0.0009,0.0002],
#        'clf__alpha': np.arange(0.0001, 0.0003, 0.0001),
#        'clf__l1_ratio': np.arange(0.15,0.3,0.05),
#        'clf_max_iter': [100],
        }

sgd_model = linear_model_evaluation(
        SGDClassifier,
        epsilon=0.001,
        search_params=sgd_search,
        random_state=11,
        scoring_func=recall_score,
        class_weight='balanced',
        )


#%%


# =============================================================================
#                         SVC (Support Vector Classifier)
# =============================================================================

svc_search = {
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [1.2,1.3,1.4],
        }

svc_model = linear_model_evaluation(
        SVC,
        C=1.3,
        kernel='linear',
        cache_size=1000,
        search_params=svc_search,
        random_state=11,
        scoring_func=recall_score
        )

#%%

# List of (string, estimator) tuples
estimators = [
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('dt', DecisionTreeClassifier()),

        ]

# Build and fit a voting classifier
clf_vote = VotingClassifier(estimators, voting='hard', n_jobs=-1)
clf_vote.fit(X_train, y_train)

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators, voting='soft', n_jobs=-1)
clf_avg.fit(X_train, y_train)

print(' |*** Averaging ***| ')
print(classification_report(y_test, clf_avg.predict(X_test)))
print(' |*** Voting ***| ')
print(classification_report(y_test, clf_vote.predict(X_test)))

#%%


#                               FINAL PREDICTIONS
# =============================================================================

# Predict on Test Set
# =============================================================================
# predict on test data
print('Predicting...')
test_df = get_numeric_data(fd_clean_test)[features]
fd_clean_test['successful_investment'] = lgr_search.best_estimator_.predict(test_df)
print('Complete!')

fd_clean_test['successful_investment'].value_counts()

#%%

#                                WRITE TO FILE
# =============================================================================

print('Writing .csv to file...')
#fd_clean_test['ID'] = fd_clean_test['ID'].astype(str)
submit_csv = fd_clean_test[['ID', 'successful_investment']]
submit_csv.to_csv('find_the_fund_submit.csv',
                  encoding='utf-8',
                  index=False
                  )

print('Finished writing csv!')


print(pd.read_csv('find_the_fund_submit.csv')['successful_investment'].value_counts())
print(y_train.value_counts())

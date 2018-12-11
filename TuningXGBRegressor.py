from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import pandas as pd

def runXGBRegressorTuning(X_train, X_test, y_train, y_test,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          initial_max_depth=[3, 5, 7, 9],
                          initial_min_child_weight=[1, 3, 5],
                          objective='reg:linear',
                          learning_rate=0.1, n_estimators=140, max_depth=5,
                          min_child_weight = 1, reg_alpha=0, reg_lambda=0,
                          gamma=0, subsample=0.8, colsample_bytree=0.8):
    # Tune max depth and min child weight - strongest bearing on model tuning
    best_score = 1000000000
    xgb_param_dict = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, reg_alpha=reg_alpha,
                          gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, objective=objective, reg_lambda=reg_lambda,
                          nthread=4, scale_pos_weight=1, seed=27)
    xgb_model = XGBRegressor(**xgb_param_dict)

    param_test1 = {
        'max_depth': initial_max_depth,
        'min_child_weight': initial_min_child_weight}

    gsearch = GridSearchCV(estimator=XGBRegressor(**xgb_param_dict),
                            param_grid=param_test1, scoring=scoring, n_jobs=4, iid=False, cv=cv)
    gsearch.fit(X_train, y_train)
    print('Best params: {}'.format(gsearch.best_params_))
    print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))

    best_score = np.sqrt(-gsearch.best_score_)
    xgb_param_dict['max_depth'] = gsearch.best_params_['max_depth']
    xgb_param_dict['min_child_depth'] = gsearch.best_params_['min_child_depth']
    xgb_model = XGBRegressor(**xgb_param_dict)

    # Decision tree to determine new search ranges if optimal solution found at limit of initial range
    if gsearch.best_params_['max_depth'] == max(initial_max_depth):
        print('Best max_depth at max limit of initial range...')
        new_initial_max_depth = range(max(initial_max_depth), max(initial_max_depth) + 6, 2)
    elif gsearch.best_params_['max_depth'] == min(initial_max_depth):
        print('Best max_depth at min limit of initial range...')
        new_initial_max_depth = range(min(initial_max_depth)-6, min(initial_max_depth), 2)
    else:
        new_initial_max_depth = initial_max_depth

    if gsearch.best_params_['min_child_weight'] == max(initial_min_child_weight):
        print('Best min_child_weight at max limit of initial range...')
        new_initial_min_child_weight = range(max(initial_min_child_weight), max(initial_min_child_weight) + 6, 2)
    elif gsearch.best_params_['min_child_weight'] == min(initial_min_child_weight):
        print('Best max_depth at min limit of initial range...')
        new_initial_min_child_weight = range(min(initial_min_child_weight)-6, min(initial_min_child_weight), 2)
    else:
        new_initial_min_child_weight= initial_min_child_weight

    # Run various procedures depending on outcome
    if new_initial_max_depth != initial_min_child_weight or  new_initial_max_depth != initial_max_depth:
        param_test = {'max_depth': new_initial_max_depth, 'min_child_weight': new_initial_min_child_weight}
        gsearch = GridSearchCV(estimator=xgb_model,
                               param_grid=param_test, scoring=scoring, n_jobs=4, iid=False, cv=cv)
        gsearch.fit(X_train, y_train)
        print('Best params: {}'.format(gsearch.best_params_))
        print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
        best_score = np.sqrt(-gsearch.best_score_)
        xgb_param_dict['max_depth'] = gsearch.best_params_['max_depth']
        xgb_param_dict['min_child_depth'] = gsearch.best_params_['min_child_depth']
        xgb_model = XGBRegressor(**xgb_param_dict)

    else:
        # Check either side of best variables to check
        param_test = {'max_depth': [xgb_param_dict['max_depth']-1, xgb_param_dict['max_depth'], xgb_param_dict['max_depth']+1],
                      'min_child_weight': [xgb_param_dict['min_child_weight']-1, xgb_param_dict['min_child_weight'], xgb_param_dict['min_child_weight']+1]}
        gsearch = GridSearchCV(estimator=xgb_model,
                               param_grid=param_test, scoring=scoring, n_jobs=4, iid=False, cv=cv)
        gsearch.fit(X_train, y_train)
        # Fine-tuned max_depth and min_child_weight parameters
        print('Fine-tuned max_depth and min_child_weight parameters...\n')
        print('Best params: {}'.format(gsearch.best_params_))
        print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
        best_score = np.sqrt(-gsearch.best_score_)
        xgb_param_dict['max_depth'] = gsearch.best_params_['max_depth']
        xgb_param_dict['min_child_weight'] = gsearch.best_params_['min_child_weight']
        xgb_model = XGBRegressor(**xgb_param_dict)

    warnings = {}
    # Tune gamma
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_test3, scoring=scoring, n_jobs=4, iid=False, cv=cv)

    gsearch.fit(X_train, y_train)
    # Fine-tuned gamma parameters
    print('Fine-tuned gamma parameters...\n')
    print('Best params: {}'.format(gsearch.best_params_))
    print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
    best_score = np.sqrt(-gsearch.best_score_)
    xgb_param_dict['gamma'] = gsearch.best_params_['gamma']
    xgb_model = XGBRegressor(**xgb_param_dict)

    if xgb_param_dict['gamma'] == max(param_test3['gamma']):
        warnings['gamma'] = 'gamma: Optimal parameter {} at max of search range'.format(xgb_param_dict['gamma'])


    # Tune subsample and colsample_bytree
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_test4, scoring=scoring, n_jobs=4, iid=False, cv=cv)

    gsearch.fit(X_train, y_train)
    # Fine-tuned subsample and colsample_bytree parameters
    print('Tuned subsample and colsample_bytree parameters...\n')
    print('Best params: {}'.format(gsearch.best_params_))
    print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
    best_score = np.sqrt(-gsearch.best_score_)
    xgb_param_dict['subsample'] = gsearch.best_params_['subsample']
    xgb_param_dict['colsample_bytree'] = gsearch.best_params_['colsample_bytree']
    xgb_model = XGBRegressor(**xgb_param_dict)

   # while xgb_param_dict['subsample'] == max(param_test4['subsample'] or
   #       xgb_param_dict['colsample_bytree'] == max(param_test4['colsample_bytree']) or
   #       xgb_param_dict['subsample'] == min(param_test4['subsample'] or
   #       xgb_param_dict['colsample_bytree'] == min(param_test4['colsample_bytree']):

    if xgb_param_dict['subsample'] == max(param_test4['subsample']):
        warnings['subsample'] = 'subsample: Optimal parameter {} at max of search range'.format(xgb_param_dict['subsample'])
    elif xgb_param_dict['subsample'] == min(param_test4['subsample']):
        warnings['subsample'] = 'subsample: Optimal parameter {} at min of search range'.format(xgb_param_dict['subsample'])

    if xgb_param_dict['colsample_bytree'] == max(param_test4['colsample_bytree']):
        warnings['colsample_bytree'] = 'colsample_bytree: Optimal parameter {} at max of search range'.format(xgb_param_dict['colsample_bytree'])
    elif xgb_param_dict['colsample_bytreee'] == min(param_test4['colsample_bytree']):
        warnings['colsample_bytree'] = 'colsample_bytree: Optimal parameter {} at min of search range'.format(xgb_param_dict['colsample_bytree'])

    # Tune regularisation parameters
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_test6, scoring=scoring, n_jobs=4, iid=False, cv=cv)

    gsearch.fit(X_train, y_train)
    # Fine-tuned regularisation parameters
    print('Tuned regularisation parameters...\n')
    print('Best params: {}'.format(gsearch.best_params_))
    print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
    best_score = np.sqrt(-gsearch.best_score_)
    xgb_param_dict['reg_alpha'] = gsearch.best_params_['reg_alpha']
    xgb_model = XGBRegressor(**xgb_param_dict)

    # Fine-tune regularisation parameters

    param_test7 = {
        'reg_alpha': [float(xgb_param_dict['reg_alpha'])/10,
                      float(xgb_param_dict['reg_alpha'])/2,
                      float(xgb_param_dict['reg_alpha']),
                      float(xgb_param_dict['reg_alpha'])*5,
                      float(xgb_param_dict['reg_alpha'])*2]
                    }

    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_test6, scoring=scoring, n_jobs=4, iid=False, cv=cv)

    gsearch.fit(X_train, y_train)
    # Fine-tuned regularisation parameters
    print('Tuned regularisation parameters...\n')
    print('Best params: {}'.format(gsearch.best_params_))
    print('Best score: {}'.format(np.sqrt(-gsearch.best_score_)))
    best_score = np.sqrt(-gsearch.best_score)
    xgb_param_dict['reg_alpha'] = gsearch.best_params_['reg_alpha']
    xgb_model = XGBRegressor(**xgb_param_dict)

    # Tune the learning rate of the model

    cvresult = xgb.cv(xgb_model.get_params(), X_train, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=cv,
                      metrics='rmse', early_stopping_rounds=50, show_progress=False)

    # Set the model to the optimal number of estimators wrt early stopping round limit
    xgb_param_dict['n_estimators'] = cvresult.shape[0]

    # Learn final XGBoost model
    xgb_model = XGBRegressor(**xgb_param_dict)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  eval_metric='rmse',
                  verbose=True)


    return xgb_model, xgb_model.get_params(), xgb_model.evals_result(), warnings

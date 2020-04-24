
class Config_svm:
    feature_funcs = {'min_var': 'np.min(variances,axis=1)',
                    'mean_var': 'np.mean(variances,axis=1)',
                 'max_var': 'np.max(variances,axis=1)',
                 'var_var': 'np.var(variances,axis=1)', 
                 'mean_means': 'np.mean(means,axis=1)', 
                 'min_means': 'np.min(means,axis=1)',
                 'max_means': 'np.max(means,axis=1)',
                 'var_means': 'np.var(means,axis=1)',
                 'max_vars': 'np.var(max_class,axis=1)'}

    feature_list = ['max_means', 'min_var']
    min_size_of_val_set = 200
    frac_val_test = 0.33
    filter = False

cfg_svm = Config_svm()

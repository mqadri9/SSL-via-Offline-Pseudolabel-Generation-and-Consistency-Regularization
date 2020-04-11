import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_loops',type=int,help='number of loops')

    args = parser.parse_args()
    for i in range(0,args.n_loops):
        test_acc = np.load('test_accuracies_ckpt_loop_{}_perc_0.5.pth.npy'.format(i))
        train_acc = np.load('train_accuracies_ckpt_loop_{}_perc_0.5.pth.npy'.format(i))
        var_cor = np.load('variances_cor_loop_{}.npy'.format(i))
        var_incor = np.load('variances_inc_loop_{}.npy'.format(i))
        print('Loop {}:'.format(i))
        print('- max train accuracy: {}'.format(np.max(train_acc)))
        print('- max test accuracy: {}'.format(np.max(test_acc)))
        print('- num of correct pseudolabels {}'.format(len(var_cor)))
        print('- num of incorrect pseudolabels {}'.format(len(var_incor)))
        print('- mean and std of variances for correct: {} +/- {}'.format(np.mean(var_cor),np.std(var_cor)))
        print('- mean and std of variances for incorrect: {} +/- {}'.format(np.mean(var_incor),np.std(var_incor)))
        

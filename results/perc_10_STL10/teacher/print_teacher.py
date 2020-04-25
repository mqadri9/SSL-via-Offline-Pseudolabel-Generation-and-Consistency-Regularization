import numpy as np

train_acc = np.load('train_accuracies_ckpt_loop_0_perc_10.pth.npy')

test_acc = np.load('test_accuracies_ckpt_loop_0_perc_10.pth.npy')

print('- max train accuracy: {}'.format(np.max(train_acc)))
print('- max test accuracy: {}'.format(np.max(test_acc)))

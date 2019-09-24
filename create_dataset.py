from utils import *


# folders with data, please unzip files
TRAIN_FOLDER = './noisy_train/'
TARGET_FOLDER = './params_train/'
TEST_FOLDER = './noisy_test/'

# save test data 
all_test, all_test_heads = data_extractor(62900, TEST_FOLDER)
np.save('data/all_test_heads.npy', all_test_heads)
np.save('data/all_test.npy', all_test)
del all_test, all_test_heads 

# save train data 
all_train, all_train_heads = data_extractor(146800, TRAIN_FOLDER)
np.save('data/all_train_heads.npy', all_train_heads)
np.save('data/all_train.npy', all_train)
del all_train_heads, all_train 

# save targets
all_targets = targets_extractor(TARGET_FOLDER)
np.save('data/all_targets.npy', all_targets)
print('Done with data preparation!')
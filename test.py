import hashlib
import os
import pickle
from urllib.request import urlretrieve
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile
print('All modules imported.')

def download(url, file):

    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
'notMNIST_train.zip file is corrupted. Remove the file and try again.'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
'notMNIST_test.zip file is corrupted. Remove the file and try again.'

print('All files downloaded.')

def uncompress_features_labels(file):

    features = []
    labels = []
    with ZipFile(file) as zipf:

        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        for filename in filenames_pbar:

            if not filename.endswith('/'): #str.endswith()

                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    feature = np.array(image, dtype=np.float32).flatten()
                label = os.path.split(filename)[1][0]
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)


train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)
is_features_normal = False
is_labels_encod = False
print('All features and labels uncompressed.')

# 对于灰度图像数据的归⼀化
def normalize_grayscale(image_data):

    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

# 如果标记为False，则归⼀化处理，并置标记为True
if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

if not is_labels_encod:
# 应⽤独热编码，将labels转化成数字（0/1表⽰）
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)
# 转化为float32的格式，便于后⾯在TensorFlow可以进⾏乘法运算
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'
# 为训练集和验证集随机选取数据
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

pickle_file = 'notMNIST.pickle'

# 保存数据⽅便调⽤
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file): #判断是否存在此⽂件，若⽆则存储
    print('Saving data to pickle file...')
    try:
        with open('notMNIST.pickle', 'wb') as pfile:
            pickle.dump(
                {
                'train_dataset': train_features,
                'train_labels': train_labels,
                'valid_dataset': valid_features,
                'valid_labels': valid_labels,
                'test_dataset': test_features,
                'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
print('Data cached in pickle file.')

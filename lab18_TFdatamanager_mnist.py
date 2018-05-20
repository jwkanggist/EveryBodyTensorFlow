import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


dataset_home_path = '/Users/jwkangmacpro/SourceCodes/datamanager_example/mnist_png/'
datatype = 'training/'
classtype = '*/'

datapath = dataset_home_path +  datatype + classtype + '*.png'
data_list = glob(datapath)

path = data_list[0]
path.split('/')[-2]


def get_label_from_path(path):
    return path.split('/')[-2]

# we can obtain the label and path of the dataset

## import label name
class_name = get_label_from_path(path=path)
path, class_name

# check !!
# rand_n = 9999
# path = data_list[rand_n]
# path, get_label_from_path(path=path)
# -------------------------------------

## import images
def read_image(path):

    image = np.array(Image.open(path))

    # specify the input channel number '1' for gray images
    return image.reshape(image.shape[0], image.shape[1],1)

mnist_image = read_image(path=path)


# show directory of interest
os.listdir(dataset_home_path + datatype)

label_name_list = []


for path in data_list:
    label_name_list.append(get_label_from_path(path=path))


unique_label_names = np.unique(label_name_list)
unique_label_names



def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label

onehot_encode_label(path= path)

read_image(path).shape, onehot_encode_label(path)

# Make batch data
mnist_image.shape

batch_size  = 64
data_height = 28
data_width  = 28
chanel_n    = 1

num_classes = 10


# method 1
batch_image = np.zeros((batch_size, data_height,data_width, chanel_n))
batch_label  = np.zeros((batch_size, num_classes))

batch_image.shape, batch_label.shape

# enumerate을 사용하면 앞에는 n 앞 뒤에는 index
for n, path in enumerate(data_list[:batch_size]):

    image = read_image(path=path)
    onehot_label = onehot_encode_label(path = path)
    batch_image[n,:,:,:] = image
    batch_label[n,:] = onehot_label


test_n = 0
plt.title(batch_label[test_n])
plt.imshow(batch_image[test_n, :,:,0])
plt.show()

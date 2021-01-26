import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn import preprocessing
from tqdm import tqdm


def rowSplit(rowString):
    split = rowString.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = str(split[2])  # tag is a string, not a float like the rating

    result = tuple(split)

    return result


data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()


ICM_file_asset = open('dataset/data_ICM_asset.csv', 'r')

ICM_file_asset.seek(14)
ICM_tuples = []

for line in ICM_file_asset:
    ICM_tuples.append(rowSplit(line))

itemList_icm_asset, userList_icm, tagList_icm_asset = zip(*ICM_tuples)


itemList_icm_asset = list(itemList_icm_asset)
tagList_icm_asset = list(tagList_icm_asset)

le = preprocessing.LabelEncoder()
le.fit(tagList_icm_asset)

tagList_icm_asset = le.transform(tagList_icm_asset)


print(min(tagList_icm_asset))
print(max(tagList_icm_asset))

n_items = URM_all.shape[1]
n_tags = max(tagList_icm_asset) + 1

ICM_shape = (n_items, n_tags)

ones_asset = np.ones(len(tagList_icm_asset)) * 0.9
ICM_all_asset_weighted = sps.coo_matrix((ones_asset, (itemList_icm_asset, tagList_icm_asset)), shape=ICM_shape)
ICM_all_asset_weighted = ICM_all_asset_weighted.tocsr()


data_sub_class = pd.read_csv('dataset/data_ICM_sub_class.csv')
itemList_icm_subclass = list(data_sub_class["row"])
subclass_list = list(data_sub_class["col"])
ones_subclass = list(data_sub_class["data"])

ones_subclass = np.array(ones_subclass) * 1.1

subclass_list = [x-1 for x in subclass_list]

print(min(subclass_list))
print(max(subclass_list))
print(len(set(subclass_list)))

ICM_all_subClass_weighted = sps.coo_matrix((ones_subclass, (itemList_icm_subclass, subclass_list)))
ICM_all_subClass_weighted = ICM_all_subClass_weighted.tocsr()


ICM_file_price = open('dataset/data_ICM_price.csv', 'r')

ICM_file_price.seek(14)
ICM_tuples = []

for line in ICM_file_price:
    ICM_tuples.append(rowSplit(line))

itemList_icm_price, userList_icm, tagList_icm_price = zip(*ICM_tuples)


itemList_icm_price = list(itemList_icm_price)
tagList_icm_price = list(tagList_icm_price)

le = preprocessing.LabelEncoder()
le.fit(tagList_icm_price)

tagList_icm_price = le.transform(tagList_icm_price)


print(min(tagList_icm_price))
print(max(tagList_icm_price))

n_items = URM_all.shape[1]
n_tags = max(tagList_icm_price) + 1

ICM_shape = (n_items, n_tags)

ones_price = np.ones(len(tagList_icm_price)) * 0.7
ICM_all_price_weighted = sps.coo_matrix((ones_price, (itemList_icm_price, tagList_icm_price)), shape=ICM_shape)
ICM_all_price_weighted = ICM_all_price_weighted.tocsr()

sps.save_npz('myFiles/ICM_asset_weighted', ICM_all_asset_weighted, compressed=True)
sps.save_npz('myFiles/ICM_subClass_weighted', ICM_all_subClass_weighted, compressed=True)
sps.save_npz('myFiles/ICM_price_weighted', ICM_all_price_weighted, compressed=True)

tagList_icm_asset = [x+2010 for x in tagList_icm_asset]
tagList_icm_price = [x+2945 for x in tagList_icm_price]

ones_subclass = list(ones_subclass)
ones_price = list(ones_price)
ones_asset = list(ones_asset)

item_features = subclass_list + tagList_icm_asset + tagList_icm_price
items_list = itemList_icm_subclass + itemList_icm_asset + itemList_icm_price
rating_ones = ones_subclass + ones_asset + ones_price


ICM_all = sps.coo_matrix((rating_ones, (items_list, item_features)), shape=(18495, 3966))
ICM_all = ICM_all.tocsr()

sps.save_npz('myFiles/ICM_all_weighted', ICM_all, compressed=True)


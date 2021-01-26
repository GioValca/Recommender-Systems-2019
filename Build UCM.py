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


UCM_file_age = open('dataset/data_UCM_age.csv', 'r')

UCM_file_age.seek(14)
UCM_tuples = []

for line in UCM_file_age:
    UCM_tuples.append(rowSplit(line))

userList_ucm_age, tagList_ucm_age, userList_ucm = zip(*UCM_tuples)


userList_ucm_age = list(userList_ucm_age)
tagList_ucm_age = list(tagList_ucm_age)


print(min(tagList_ucm_age))
print(max(tagList_ucm_age))

n_users = URM_all.shape[0]
n_tags = max(tagList_ucm_age) + 1

UCM_shape = (n_users, n_tags)

ones = np.ones(len(tagList_ucm_age))
UCM_all_age = sps.coo_matrix((ones, (userList_ucm_age, tagList_ucm_age)), shape=UCM_shape)
UCM_all_age = UCM_all_age.tocsr()


UCM_file_region = open('dataset/data_UCM_region.csv', 'r')

UCM_file_region.seek(14)
UCM_tuples = []

for line in UCM_file_region:
    UCM_tuples.append(rowSplit(line))

userList_ucm_region, tagList_ucm_region, userList_ucm = zip(*UCM_tuples)


userList_ucm_region = list(userList_ucm_region)
tagList_ucm_region = list(tagList_ucm_region)


print(min(tagList_ucm_region))
print(max(tagList_ucm_region))

n_users = URM_all.shape[0]
n_tags = max(tagList_ucm_region) + 1

UCM_shape = (n_users, n_tags)

ones = np.ones(len(tagList_ucm_region))
UCM_all_region = sps.coo_matrix((ones, (userList_ucm_region, tagList_ucm_region)), shape=UCM_shape)
UCM_all_region = UCM_all_region.tocsr()

cluster = pd.read_csv('myFiles/cluster.csv')
userList = [x for x in range(0, 30911)]
cluster_n = list(cluster["interaction_class"])
ones = np.ones(30911)

UCM_all_interactions = sps.coo_matrix((ones, (userList, cluster_n)))
UCM_all_interactions = UCM_all_interactions.tocsr()


sps.save_npz('myFiles/UCM_age', UCM_all_age, compressed=True)
sps.save_npz('myFiles/UCM_region', UCM_all_region, compressed=True)
sps.save_npz('myFiles/UCM_interactions', UCM_all_interactions, compressed=True)

tagList_ucm_region = [x+11 for x in tagList_ucm_region]
cluster_n = [x+17 for x in cluster_n]

user_features = tagList_ucm_age + tagList_ucm_region + cluster_n
user_list = userList_ucm_age + userList_ucm_region + userList
rating_ones = np.ones(len(user_list))


UCM_all = sps.coo_matrix((rating_ones, (user_list, user_features)), shape=(30911, 21))
UCM_all = UCM_all.tocsr()

sps.save_npz('myFiles/UCM_all', UCM_all, compressed=True)
import random

import pandas as pd
import scipy.sparse as sps

data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library

numberInteractions = 0

for _ in data["row"]:
    numberInteractions += 1

print("The number of total interactions is: ", numberInteractions)

userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

userList_unique = list(set(userList))
itemList_unique = list(set(itemList))

numUsers = len(userList_unique)
numItems = len(itemList_unique)

print("The length of the user list is ", len(userList))
print("The length of the item list is ", len(itemList))
print("The length of the unique user list is ", len(userList_unique))
print("The length of the unique item list is ", len(itemList_unique))

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all.tocsr()

nnzPerRow = URM_all.getnnz(axis=1)


outputFile = open('myFiles/NnzElementPerRow.txt', 'w')

for element in nnzPerRow:
    print(element, file=outputFile)
outputFile.close()

outputFile = open('myFiles/NnzSum.txt', 'w')

sum = 0
nnzsum = []

for element in nnzPerRow:
    sum = sum + element
    nnzsum.append(sum)
    print(sum, file=outputFile)
outputFile.close()

train_Matrix = sps.csr_matrix(URM_all)
test_users = []
test_items = []
test_ratings = []

for x in range(0, 30911):
    if nnzPerRow[x] != 0:
        if x == 0:
            r = random.randint(0, nnzPerRow[0])
        else:
            r = random.randint(nnzsum[x-1], nnzsum[x]-1)
        itemToBeRemoved = itemList[r]
        train_Matrix[x, itemToBeRemoved] = 0
        test_users.append(x)
        test_items.append(itemToBeRemoved)
        test_ratings.append(1.0)

train_Matrix.eliminate_zeros()
#NOW TRAIN_MATRIX CONTAINS THE TRAIN SET

test_Matrix = sps.coo_matrix((test_ratings, (test_users, test_items)), shape=(30911, 18495))
test_Matrix.tocsr()
#NOW TEST_MATRIX CONTAINS THE TEST SET

#HERE I SAVE THE TRAIN AND TEST SET INTO .npz FILE
sps.save_npz("train_set", train_Matrix, compressed=True)
sps.save_npz("test_set", test_Matrix, compressed=True)






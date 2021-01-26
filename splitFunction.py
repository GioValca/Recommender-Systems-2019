import scipy.sparse as sps
import random


def splitURM(URM_all):

    itemList = URM_all.nonzero()[1]     #get the indices of columns
    nnzPerRow = URM_all.getnnz(axis=1)

    sum = 0
    nnzsum = []

    for element in nnzPerRow:
        sum = sum + element
        nnzsum.append(sum)

    train_Matrix = sps.csr_matrix(URM_all)
    test_users = []
    test_items = []
    test_ratings = []

    for x in range(0, 30911):
        if nnzPerRow[x] != 0:
            if x == 0:
                r = random.randint(0, nnzPerRow[0])
            else:
                r = random.randint(nnzsum[x - 1], nnzsum[x] - 1)
            itemToBeRemoved = itemList[r]
            train_Matrix[x, itemToBeRemoved] = 0
            test_users.append(x)
            test_items.append(itemToBeRemoved)
            test_ratings.append(1.0)

    train_Matrix.eliminate_zeros()
    train_Matrix.tocsr()
    # NOW TRAIN_MATRIX CONTAINS THE TRAIN SET

    test_Matrix = sps.coo_matrix((test_ratings, (test_users, test_items)), shape=(30911, 18495))
    test_Matrix.tocsr()
    # NOW TEST_MATRIX CONTAINS THE TEST SET

    return train_Matrix, test_Matrix

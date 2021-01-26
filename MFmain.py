from matrix_factor_model import ProductRecommender
import scipy.sparse as sps
import pandas as pd

dataset = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(dataset["row"])  # set of users that have a rating for at least one item
itemList = list(dataset["col"])  # set of items that were rated by at least one user
ratingList = list(dataset["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()
nnz = URM_all.getnnz(axis=1)

data = []
begin = 0
end = 0

for i in range(30911):
    tempValues = []
    end = end + nnz[i]
    for x in range(begin, end):
        tempValues.append(itemList[x])
    begin = end
    data.append(tempValues)
print(data)

modelA = ProductRecommender()
modelA.fit(data)

# predict for user 2
print(modelA.predict_instance(0))

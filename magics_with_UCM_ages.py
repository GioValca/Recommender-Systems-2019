import pandas as pd
import scipy.sparse as sps
from Algorithms import TopPopRecommender

data = pd.read_csv('dataset/data_UCM_age.csv')  # reading the file with pandas library
userList = list(data["row"])
ageList = list(data["col"])
ratingList = list(data["data"])

age0 = []
age1 = []
age2 = []
age3 = []
age4 = []
age5 = []
age6 = []
age7 = []
age8 = []
age9 = []
age10 = []

for i in range(len(userList)):
    if ageList[i] == 0:
        age0.append(userList[i])
    if ageList[i] == 1:
        age1.append(userList[i])
    if ageList[i] == 2:
        age2.append(userList[i])
    if ageList[i] == 3:
        age3.append(userList[i])
    if ageList[i] == 4:
        age4.append(userList[i])
    if ageList[i] == 5:
        age5.append(userList[i])
    if ageList[i] == 6:
        age6.append(userList[i])
    if ageList[i] == 7:
        age7.append(userList[i])
    if ageList[i] == 8:
        age8.append(userList[i])
    if ageList[i] == 9:
        age9.append(userList[i])
    if ageList[i] == 10:
        age10.append(userList[i])

data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()
matrix = sps.csr_matrix(URM_all)

# urm0 = matrix[age0, :]
urm1 = matrix[age1, :]
urm2 = matrix[age2, :]
urm3 = matrix[age3, :]
urm4 = matrix[age4, :]
urm5 = matrix[age5, :]
urm6 = matrix[age6, :]
urm7 = matrix[age7, :]
urm8 = matrix[age8, :]
urm9 = matrix[age9, :]
urm10 = matrix[age10, :]

urm_list = [urm1, urm2, urm3, urm4, urm5, urm6, urm7, urm8, urm9, urm10]
age_list = [age1, age2, age3, age4, age5, age6, age7, age8, age9, age10]


def get_age_list():
    return age_list


j = 0

for i in urm_list:
    recommenderTP = TopPopRecommender()
    recommenderTP.fit(i)
    filename = 'TopPopForAge' + '{}'.format(j+1)
    file = open('myFiles/' + filename + '.txt', 'w')
    items_rec = recommenderTP.recommend(age_list[j][0], at=10)
    sarr = [str(a) for a in items_rec]
    print(' '.join(sarr), file=file)
    file.close()
    j += 1

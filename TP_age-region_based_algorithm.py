import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from Algorithms import TopPopRecommender


data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()
matrix = sps.csr_matrix(URM_all)
read_file_name = 'region_user_combination/age'
urm_list = []
list_of_users_per_combination = []

for age in range(1, 11):
    for region in range(2, 8):
        read_file_name = 'region_user_combination/age'
        read_file_name = read_file_name + '{}'.format(age) + 'region' + '{}'.format(region) + '.csv'
        df_users = pd.read_csv(read_file_name)
        users_list = list(df_users['users'])
        print('lunghezza user list è di {}'.format(len(users_list)))
        print(users_list)
        list_of_users_per_combination.append(users_list)
        urm_new = matrix[users_list, :]
        urm_list.append(urm_new)

a = 0
r = 2
j = 0

for i in tqdm(urm_list):
    recommenderTP = TopPopRecommender()
    recommenderTP.fit(i)
    filename = 'TopPopForAge' + '{}'.format(a+1) + 'AndRegion' + '{}'.format(r)
    file = open('myCombinations/' + filename + '.txt', 'w')
    items_rec = recommenderTP.recommend(list_of_users_per_combination[j][0], at=10)
    sarr = [str(a) for a in items_rec]
    print(' '.join(sarr), file=file)
    file.close()
    r += 1
    if r == 8:
        a += 1
        r = 2
    j += 1

import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from Algorithms import TopPopRecommender


data = pd.read_csv('dataset/data_UCM_region.csv')
userlist = list(data["row"])
regionList = list(data["col"])
ratingList = list(data["data"])
df_region = data.set_index("row", drop=False)

#print(set(regionList))

#print(userlist)

region0 = []
region2 = []
region3 = []
region4 = []
region5 = []
region6 = []
region7 = []

for i in range(len(userlist)):
    if regionList[i] == 0:
        region0.append(userlist[i])
    if regionList[i] == 2:
        region2.append(userlist[i])
    if regionList[i] == 3:
        region3.append(userlist[i])
    if regionList[i] == 4:
        region4.append(userlist[i])
    if regionList[i] == 5:
        region5.append(userlist[i])
    if regionList[i] == 6:
        region6.append(userlist[i])
    if regionList[i] == 7:
        region7.append(userlist[i])


def number_of_people_in_region(region_id):
    if region_id == 0:
        return len(region0)
    if region_id == 2:
        return len(region2)
    if region_id == 3:
        return len(region3)
    if region_id == 4:
        return len(region4)
    if region_id == 5:
        return len(region5)
    if region_id == 6:
        return len(region6)
    if region_id == 7:
        return len(region7)

'''
print('The number of users who live in region 0 is: {}'.format(len(region0)))
print('The number of users who live in region 2 is: {}'.format(len(region2)))
print('The number of users who live in region 3 is: {}'.format(len(region3)))
print('The number of users who live in region 4 is: {}'.format(len(region4)))
print('The number of users who live in region 5 is: {}'.format(len(region5)))
print('The number of users who live in region 6 is: {}'.format(len(region6)))
print('The number of users who live in region 7 is: {}'.format(len(region7)))
'''



datab = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(datab["row"])  # set of users that have a rating for at least one item
itemList = list(datab["col"])  # set of items that were rated by at least one user
ratingList = list(datab["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()
matrix = sps.csr_matrix(URM_all)

urm0 = matrix[region0, :]
urm2 = matrix[region2, :]
urm3 = matrix[region3, :]
urm4 = matrix[region4, :]
urm5 = matrix[region5, :]
urm6 = matrix[region6, :]
urm7 = matrix[region7, :]


urm_list = [urm0, urm2, urm3, urm4, urm5, urm6, urm7]
region_list = [region0, region2, region3, region4, region5, region6, region7]

'''
#CODE TO WRITE USERS WITH MORE REGIONS FILE
temp = userlist

for x in set(temp):
    temp.remove(x)

li = list(set(temp))
li.sort()

output = open('myFiles/users_with_more_regions.csv', 'w')
print('user_id', file=output)

for i in li:
    print(i, file=output)
output.close()
print(len(li))
'''

j = 0

'''''
for i in tqdm(urm_list):
    recommenderTP = TopPopRecommender()
    recommenderTP.fit(i)
    filename = 'TopPopForRegion' + '{}'.format(j+1)
    file = open('myFiles/' + filename + '.txt', 'w')
    items_rec = recommenderTP.recommend(region_list[j][0], at=10)
    sarr = [str(a) for a in items_rec]
    print(' '.join(sarr), file=file)
    file.close()
    j += 1
'''


def choose_region(list_of_regions_of_user):
    if list_of_regions_of_user[0] > list_of_regions_of_user[1]:
        return list_of_regions_of_user[0]
    else:
        return list_of_regions_of_user[1]

def get_users_with_region():
    return userlist, region_list

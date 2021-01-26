import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from Algorithms import TopPopRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from printCsvFunction import print_to_csv_age_and_region



data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()

recommenderICF = ItemKNNCFRecommender(URM_all)
recommenderICF.fit(shrink=24, topK=10)

recommenderTP = TopPopRecommender()
recommenderTP.fit(URM_all)

filename = 'out' + '59'
# REMEMBER TO CHANGE THE FILE NAME!!!
print_to_csv_age_and_region(recommenderICF, recommenderTP, filename)

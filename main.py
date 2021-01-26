import pandas as pd
import scipy.sparse as sps
from matplotlib import pyplot
from tqdm import tqdm


from Base.Evaluation.Evaluator import EvaluatorHoldout
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from printCsvFunction import print_to_csv_age
from printCsvFunction import print_to_csv_region
from printCsvFunction import print_to_csv_age_and_region
from Notebooks_utils.evaluation_function import evaluate_algorithm
from splitFunction import splitURM


data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()

'''
warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

URM_all = URM_all[:, warm_items]

warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

URM_all = URM_all[warm_users, :]

print(URM_all)
'''

# URM_train, URM_test = splitURM(URM_all)
ICM_all = sps.load_npz('myFiles/ICM_all.npz')
URM_train = sps.load_npz('myFiles/Train_for_main.npz')
URM_test = sps.load_npz('myFiles/Test_for_main.npz')
# sps.save_npz('myFiles/URM_all.npz', URM_all, compressed=True)
# sps.save_npz('MyFiles/Train_for_main.npz', URM_train, compressed=True)
# sps.save_npz('MyFiles/Test_for_main.npz', URM_test, compressed=True)
URM_train = URM_train.tocsr()
URM_test = URM_test.tocsr()



# recommenderICF = ItemKNNCFRecommender(URM_train)
# recommenderICF.fit(shrink=30, topK=10, similarity='jaccard', normalize=True)

recommenderCB = ItemKNNCBFRecommender(URM_train, ICM_all)
recommenderCB.fit(shrink=112.2, topK=5, normalize=True, similarity='jaccard')

# recommenderTP = TopPopRecommender()
# recommenderTP.fit(URM_train)

# recommenderGRAPH = P3alphaRecommender(URM_train)
# recommenderGRAPH.fit(topK=10, alpha=0.22, implicit=True, normalize_similarity=True)

# recommenderBetaGRAPH = RP3betaRecommender(URM_train)
# recommenderBetaGRAPH.fit(topK=54, implicit=True, normalize_similarity=True, alpha=1e-6, beta=0.2, min_rating=0)

# recommenderSLIMELASTIC = SLIMElasticNetRecommender(URM_all)
# recommenderSLIMELASTIC.fit(topK=10, alpha=1e-4)
# recommenderSLIMELASTIC.save_model('model/', file_name='SLIM_ElasticNet')

# recommenderCYTHON = SLIM_BPR_Cython(URM_train, recompile_cython=False)
# recommenderCYTHON.fit(epochs=350, batch_size=200, sgd_mode='adagrad', learning_rate=0.001, topK=10)

# URM_validation = sps.load_npz('URM/URM_validation.npz')

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

result, result_string = evaluator_test.evaluateRecommender(recommenderCB)
print(result_string)

# evaluate_algorithm(URM_test, recommenderGRAPH, recommenderTP, at=10)

'''
x_tick = [x for x in range(15, 28, 3)]
MAP_per_k = []
trains = []
tests = []

for i in range(4):
    URM_train, URM_test = splitURM(URM_all)
    trains.append(URM_train)
    tests.append(URM_test)

i = 0

for train in tqdm(trains):
    for shrink in x_tick:

        recommender = UserKNNCFRecommender(train)
        recommender.fit(shrink=shrink, topK=10)

        recommenderTP = TopPopRecommender()
        recommenderTP.fit(train)

        result_dict = evaluate_algorithm(tests[i], recommender, recommenderTP, at=10)
        MAP_per_k.append(result_dict["MAP"])

    i = i + 1
    print("For train {}".format(i) + " MAPs are: " + "{}".format(MAP_per_k))


pyplot.plot(x_tick, MAP_per_k)
pyplot.ylabel('MAP')
pyplot.xlabel('shrink')
pyplot.show()

#result_dict = evaluate_algorithm(URM_test, recommender, recommenderTP, at=10)



# filename = 'out' + '24'
# REMEMBER TO CHANGE THE FILE NAME!!!
# print_to_csv_age_and_region(recommenderSLIMELASTIC, recommenderTP, filename)
'''


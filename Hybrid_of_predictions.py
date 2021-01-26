from sklearn import preprocessing

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from tqdm import tqdm
import random
import pandas as pd
import scipy.sparse as sps
from Algorithms import ItemCFKNNRecommender, TopPopRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMetrics
from printCsvFunction import print_to_csv_age_and_region
from splitFunction import splitURM
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender, BaseItemSimilarityMatrixRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from MatrixFactorization.IALSRecommender import IALSRecommender


class ItemKNNScoresHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, Recommender_4, Recommender_5,
                 Recommender_6, Recommender_7):
        super(ItemKNNScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.Recommender_4 = Recommender_4
        self.Recommender_5 = Recommender_5
        self.Recommender_6 = Recommender_6
        self.Recommender_7 = Recommender_7

        self.maximums = []

    def fit(self, alpha=0.5, beta=0.3, gamma=0.2, delta=0.1, epsilon=0.1, zeta=0.1, eta=0.1, normalize=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.normalize = normalize

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_prenormalisation = []
        item_weights_ItemCFKNN = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_SLIMElasticNet = self.Recommender_2._compute_item_score(user_id_array)
        item_weights_ItemCBF = self.Recommender_3._compute_item_score(user_id_array)
        item_weights_RP3beta = self.Recommender_4._compute_item_score(user_id_array)
        item_weights_UserCFKNN = self.Recommender_5._compute_item_score(user_id_array)
        item_weights_SLIMCython = self.Recommender_6._compute_item_score(user_id_array)
        item_weights_IALS = self.Recommender_7._compute_item_score(user_id_array)

        item_weights_prenormalisation.append(item_weights_ItemCFKNN)
        item_weights_prenormalisation.append(item_weights_SLIMElasticNet)
        item_weights_prenormalisation.append(item_weights_ItemCBF)
        item_weights_prenormalisation.append(item_weights_RP3beta)
        item_weights_prenormalisation.append(item_weights_UserCFKNN)
        item_weights_prenormalisation.append(item_weights_SLIMCython)
        item_weights_prenormalisation.append(item_weights_IALS)

        if self.normalize == True:
            norm_scores = []
            for s in item_weights_prenormalisation:
                norm_scores.append(preprocessing.normalize(s))

            item_weights_ItemCFKNN = norm_scores[0]
            item_weights_SLIMElasticNet = norm_scores[1]
            item_weights_ItemCBF = norm_scores[2]
            item_weights_RP3beta = norm_scores[3]
            item_weights_UserCFKNN = norm_scores[4]
            item_weights_SLIMCython = norm_scores[5]
            item_weights_IALS = norm_scores[6]

        item_weights = item_weights_ItemCFKNN * self.alpha + item_weights_SLIMElasticNet * self.beta + \
                       item_weights_ItemCBF * self.gamma + item_weights_RP3beta * self.delta + \
                       item_weights_UserCFKNN * self.epsilon + item_weights_SLIMCython * self.zeta + \
                       item_weights_IALS * self.eta

        self.maximums.append(item_weights_ItemCBF.max())

        return item_weights


ICM_all = sps.load_npz('myFiles/ICM_all.npz')
ICM_all_weighted = sps.load_npz('myfiles/ICM_all_weighted.npz')
UCM_all = sps.load_npz('myFiles/UCM_all.npz')

print(ICM_all_weighted)

data = pd.read_csv('dataset/data_train.csv')  # reading the file data train with pandas library
userList = list(data["row"])  # set of users that have a rating for at least one item
itemList = list(data["col"])  # set of items that were rated by at least one user
ratingList = list(data["data"])

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()
URM_train = sps.load_npz('URM/URM_train_for_tuning.npz')
URM_test = sps.load_npz('URM/URM_test_for_tuning.npz')
# URM_validation = sps.load_npz('URM/URM_validation.npz')

# sps.save_npz('URM/URM_train_for_tuning.npz', URM_train, compressed=True)
# sps.save_npz('URM/URM_test_for_tuning.npz', URM_test, compressed=True)

URM_train = URM_train.tocsr()
URM_test = URM_test.tocsr()
# URM_validation = URM_validation.tocsr()

# sps.save_npz('URM/URM_test.npz', URM_test, compressed=True)
# sps.save_npz('URM/URM_train.npz', URM_train, compressed=True)
# sps.save_npz('URM/URM_validation.npz', URM_validation, compressed=True)

# evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])

# crea le matrici di raccomandazioni
recommenderTP = TopPopRecommender()
recommenderTP.fit(URM_all)

itemKNNCF = ItemKNNCFRecommender(URM_all)
itemKNNCF.fit(shrink=30, topK=10, similarity='jaccard', normalize=True)

recommenderCYTHON = SLIM_BPR_Cython(URM_all, recompile_cython=False)
# recommenderCYTHON.fit(epochs=200, batch_size=1000, sgd_mode='adagrad', learning_rate=1e-4, topK=10)
# recommenderCYTHON.save_model('model/', file_name='SLIM_Cython_max')
recommenderCYTHON.load_model('model/', file_name='SLIM_Cython_300_Ad')

recommenderCB = ItemKNNCBFRecommender(URM_all, ICM_all_weighted)
recommenderCB.fit(shrink=115, topK=10, normalize=True, similarity='jaccard')

recommenderELASTIC = SLIMElasticNetRecommender(URM_all)
# recommenderELASTIC.fit(topK=100, alpha=1e-4, positive_only=True, l1_ratio=0.5)
# recommenderELASTIC.save_model('model/', file_name='SLIM_ElasticNet_max')
recommenderELASTIC.load_model('model/', file_name='SLIM_ElasticNet_l1ratio_0_5')

# recommenderAlphaGRAPH = P3alphaRecommender(URM_all)
# recommenderAlphaGRAPH.fit(topK=10, alpha=0.41, implicit=True, normalize_similarity=True)

recommenderBetaGRAPH = RP3betaRecommender(URM_all)
recommenderBetaGRAPH.fit(topK=54, implicit=True, normalize_similarity=True, alpha=1e-6, beta=0.2, min_rating=0)

recommenderUserKNN = UserKNNCFRecommender(URM_all)
recommenderUserKNN.fit(topK=550, shrink=0, similarity='jaccard', normalize=True)

recommenderIALS = IALSRecommender(URM_all)
# recommenderIALS.fit(epochs=200, alpha=1, reg=1e-4)
# recommenderIALS.save_model('model/', file_name='IALS_1_200_max')
recommenderIALS.load_model('model/', file_name='IALS_1_200_max')

recommenderUserCBF = UserKNNCBFRecommender(URM_all, UCM_all)
recommenderUserCBF.fit(topK=3000, shrink=10, similarity='cosine', normalize=True)

hybridRecommender = ItemKNNScoresHybridRecommender(URM_all, itemKNNCF, recommenderELASTIC, recommenderCB,
                                                   recommenderBetaGRAPH, recommenderUserKNN, recommenderCYTHON,
                                                   recommenderIALS)

hybridRecommender_semiCold = ItemKNNScoresHybridRecommender(URM_all, itemKNNCF, recommenderELASTIC, recommenderCB,
                                                            recommenderBetaGRAPH, recommenderUserKNN, recommenderCYTHON,
                                                            recommenderIALS)

hybridRecommender.fit(alpha=8, beta=8.5, gamma=6, delta=8, epsilon=0.7, zeta=3, eta=0, normalize=False)
hybridRecommender_semiCold.fit(alpha=8, beta=8, gamma=40, delta=8, epsilon=0.7, zeta=3, eta=0, normalize=False)


filename = 'out' + '100'
# REMEMBER TO CHANGE THE FILE NAME!!!
print_to_csv_age_and_region(recommenderHybrid=hybridRecommender, recommenderTP=recommenderTP,
                            recommenderHybridSemiCold=hybridRecommender_semiCold, recommenderUserCBF=recommenderUserCBF,
                            filename=filename)

max = hybridRecommender.maximums

print(max)


# result, result_string = evaluator_test.evaluateRecommender(hybridRecommender)
# print(result_string)

# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# {'target': 0.05048131550057795, 'params': {'alpha': 3.7320506197325742, 'beta': 4.393344601526925, 'delta': 2.88598538296003, 'epsilon': 1.1201182924954116, 'eta': 0.9005291646412666, 'gamma': 2.9464936675713083, 'zeta': 1.5281198715810185}}
# Elapsed time = 114.31117238998414 minutes
# BEST PARAMETERS WITH NORMALISATION


'''
possible_alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
possible_beta = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
possible_gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
possible_delta = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
possible_epsilon = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
possible_zeta = [0.3, 0.4, 0.5, 0.6, 0.7]

best_config = [0.7, 0.6, 0.3, 0.5, 0.1, 0.4]
best_MAP = 0.0
current_MAP = 0.0
recommenders = [itemKNNCF.RECOMMENDER_NAME, recommenderELASTIC.RECOMMENDER_NAME, recommenderCB.RECOMMENDER_NAME,
                recommenderBetaGRAPH.RECOMMENDER_NAME, recommenderUserKNN.RECOMMENDER_NAME, recommenderCYTHON.RECOMMENDER_NAME]

log_file = open('log/parameter_tuning_hybrid_weights_Ad_2.txt', mode='w')
print("Parameters tuning log for the hybrid recommender of predictions (second version). \n" +
      "This are the recommenders that have been put inside: {}".format(recommenders), file=log_file)


print("\n", file=log_file)


for iteration in tqdm(range(400)):
    a = random.choice(possible_alpha)
    b = random.choice(possible_beta)
    c = random.choice(possible_gamma)
    d = random.choice(possible_delta)
    e = random.choice(possible_epsilon)
    f = random.choice(possible_zeta)

    current_config = [a, b, c, d, e, f]

    hybridRecommender = ItemKNNScoresHybridRecommender(URM_train, itemKNNCF, recommenderELASTIC, recommenderCB,
                                                       recommenderBetaGRAPH, recommenderUserKNN, recommenderCYTHON)

    hybridRecommender.fit(alpha=a, beta=b, gamma=c, delta=d, epsilon=e, zeta=f)

    result, result_string = evaluator_validation.evaluateRecommender(hybridRecommender)
    results_run_current_cutoff = result[10]
    for metric in results_run_current_cutoff.keys():
        if metric == "MAP":
            current_MAP = results_run_current_cutoff[metric]

    print("Iteration {}. ".format(iteration) + "Result MAP=%f" % current_MAP +
          "  Configuration tried: {}.".format(current_config), file=log_file)

    if current_MAP > best_MAP:
        best_MAP = current_MAP
        best_config = current_config
        print("\n", file=log_file)
        print("Found a better config at iteration %d, with MAP:%f" % (iteration, current_MAP), file=log_file)
        print("The new best config is: {}".format(best_config), file=log_file)
        print("\n", file=log_file)


print("\n\nFINAL RESULT\nBest config is " + "{}".format(best_config) +
      " with an incredible MAP of %f" % best_MAP, file=log_file)
'''

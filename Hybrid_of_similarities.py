# Define parameters range
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
import os
import scipy.sparse as sps
from Base.DataIO import DataIO
from Algorithms import ItemCFKNNRecommender, TopPopRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Base.Evaluation.Evaluator import EvaluatorHoldout
from printCsvFunction import print_to_csv_age_and_region
from splitFunction import splitURM
from RecSysFramework.skopt.space import Real, Integer, Categorical
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender, BaseItemSimilarityMatrixRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

'''
URM_all = sps.load_npz('myFiles/URM_all.npz')

URM_train_temp, URM_test = splitURM(URM_all)
URM_train, URM_validation = splitURM(URM_train_temp)


URM_train = URM_train.tocsr()
URM_test = URM_test.tocsr()
URM_validation = URM_validation.tocsr()

sps.save_npz(matrix=URM_train, file='URM/URM_train.npz', compressed=True)
sps.save_npz(matrix=URM_test, file='URM/URM_test.npz', compressed=True)
sps.save_npz(matrix=URM_validation, file='URM/URM_validation.npz', compressed=True)
'''

URM_train = sps.load_npz('URM/URM_train.npz')
URM_test = sps.load_npz('URM/URM_test.npz')
URM_validation = sps.load_npz('URM/URM_validation.npz')

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])

# Create BayesianSearch object
'''
recommender_class = ItemKNNCFRecommender

parameterSearch = SearchBayesianSkopt(recommender_class,
                                      evaluator_validation=evaluator_validation,
                                      evaluator_test=evaluator_test)

hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["topK"] = Integer(5, 15)
hyperparameters_range_dictionary["shrink"] = Integer(10, 50)
hyperparameters_range_dictionary["similarity"] = Categorical(["jaccard"])
hyperparameters_range_dictionary["normalize"] = Categorical([True, False])

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)
output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# RUN
n_cases = 10
metric_to_optimize = "MAP"
parameterSearch.search(recommender_input_args,
                       parameter_search_space=hyperparameters_range_dictionary,
                       n_cases=n_cases,
                       n_random_starts=5,
                       save_model="no",
                       output_folder_path=output_folder_path,
                       output_file_name_root=recommender_class.RECOMMENDER_NAME,
                       metric_to_optimize=metric_to_optimize
                       )

data_loader = DataIO(folder_path=output_folder_path)
search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata")
best_parameters = search_metadata["hyperparameters_best"]

print(best_parameters)
'''

ICM_all = sps.load_npz('myFiles/ICM_all.npz')
URM_all = sps.load_npz('myFiles/URM_all.npz')

# crea le matrici di raccomandazioni
recommenderTP = TopPopRecommender()
recommenderTP.fit(URM_all)

itemKNNCF = ItemKNNCFRecommender(URM_all)
itemKNNCF.fit(shrink=24, topK=10)

# recommenderCYTHON = SLIM_BPR_Cython(URM_all, recompile_cython=False)
# recommenderCYTHON.fit(epochs=2000, batch_size=100, sgd_mode='sdg', learning_rate=1e-5, topK=10)

# recommenderCB = ItemKNNCBFRecommender(URM_all, ICM_all)
# recommenderCB.fit(shrink=24, topK=10)

recommenderELASTIC = SLIMElasticNetRecommender(URM_all)
# recommenderELASTIC.fit(topK=10)
#recommenderELASTIC.save_model('model/', file_name='SLIM_ElasticNet')
recommenderELASTIC.load_model('model/', file_name='SLIM_ElasticNet')

recommenderAlphaGRAPH = P3alphaRecommender(URM_all)
recommenderAlphaGRAPH.fit(topK=10, alpha=0.22, implicit=True, normalize_similarity=True)

recommenderBetaGRAPH = RP3betaRecommender(URM_all)
recommenderBetaGRAPH.fit(topK=10, implicit=True, normalize_similarity=True, alpha=1e-6, beta=0.2)


class ItemKNNSimilarityHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"

    def __init__(self, URM_train, Similarity_1, Similarity_2, Similarity_3, Similarity_4, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__(URM_train)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                    Similarity_1.shape, Similarity_2.shape
                ))

        if Similarity_1.shape != Similarity_3.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S3 is {}".format(
                    Similarity_1.shape, Similarity_3.shape
                ))

        if Similarity_1.shape != Similarity_4.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S4 is {}".format(
                    Similarity_1.shape, Similarity_4.shape
                ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')
        self.Similarity_4 = check_matrix(Similarity_4.copy(), 'csr')

    def fit(self, topK=10, alpha=0.5, beta=0.3, gamma=0.2, delta=0.1):
        self.topK = topK
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        W = self.Similarity_1 * self.alpha + self.Similarity_2 * self.beta + self.Similarity_3 * self.gamma + \
            self.Similarity_4 * self.delta
        self.W_sparse = similarityMatrixTopK(W, k=self.topK).tocsr()


URM_all = sps.load_npz('myFiles/URM_all.npz')

hybridrecommender = ItemKNNSimilarityHybridRecommender(URM_all, itemKNNCF.W_sparse, recommenderELASTIC.W_sparse,
                                                       recommenderAlphaGRAPH.W_sparse, recommenderBetaGRAPH.W_sparse)
hybridrecommender.fit(alpha=0.65, beta=0.45, gamma=0.10, delta=0.10, topK=10)

# result, result_string = evaluator_validation.evaluateRecommender(hybridrecommender)
# print(result_string)


filename = 'out' + '47'
# REMEMBER TO CHANGE THE FILE NAME!!!
print_to_csv_age_and_region(hybridrecommender, recommenderTP, filename)

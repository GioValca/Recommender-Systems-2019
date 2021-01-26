import os

from Base.DataIO import DataIO
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from RecSysFramework.skopt.space import Real, Integer, Categorical
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from tqdm import tqdm
import random
import scipy.sparse as sps
from Algorithms import ItemCFKNNRecommender, TopPopRecommender
from Hybrid_of_predictions import ItemKNNScoresHybridRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMetrics
from printCsvFunction import print_to_csv_age_and_region
from splitFunction import splitURM
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender, BaseItemSimilarityMatrixRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

URM_all = sps.load_npz('myFiles/URM_all.npz')

for i in range(12):

    URM_all = sps.load_npz('myFiles/URM_all.npz')

    URM_train_temp, URM_test = splitURM(URM_all)
    URM_train, URM_validation = splitURM(URM_train_temp)

    URM_train = URM_train.tocsr()
    URM_test = URM_test.tocsr()
    URM_validation = URM_validation.tocsr()

    ICM_all = sps.load_npz('myFiles/ICM_all.npz')

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    itemKNNCF = ItemKNNCFRecommender(URM_train)
    itemKNNCF.fit(shrink=24, topK=10)

    recommenderCYTHON = SLIM_BPR_Cython(URM_train, recompile_cython=False)
    recommenderCYTHON.fit(epochs=2000, batch_size=200, sgd_mode='sdg', learning_rate=1e-5, topK=10)

    recommenderCB = ItemKNNCBFRecommender(URM_train, ICM_all)
    recommenderCB.fit(shrink=24, topK=10)

    recommenderELASTIC = SLIMElasticNetRecommender(URM_train)
    # recommenderELASTIC.fit(topK=10)
    # recommenderELASTIC.save_model('model/', file_name='SLIM_ElasticNet')
    recommenderELASTIC.load_model('model/', file_name='SLIM_ElasticNet_train')

    # recommenderAlphaGRAPH = P3alphaRecommender(URM_train)
    # recommenderAlphaGRAPH.fit(topK=10, alpha=0.22, implicit=True, normalize_similarity=True)

    recommenderBetaGRAPH = RP3betaRecommender(URM_train)
    recommenderBetaGRAPH.fit(topK=10, implicit=True, normalize_similarity=True, alpha=0.41, beta=0.049)

    recommederUserKNN = UserKNNCFRecommender(URM_train)
    recommederUserKNN.fit(topK=10, shrink=15, similarity='jaccard')

    # Create BayesianSearch object
    recommender_class = ItemKNNScoresHybridRecommender

    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    # weight_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    weight_list = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["alpha"] = Categorical(weight_list)
    hyperparameters_range_dictionary["beta"] = Categorical(weight_list)
    hyperparameters_range_dictionary["gamma"] = Categorical(weight_list)
    hyperparameters_range_dictionary["delta"] = Categorical(weight_list)
    hyperparameters_range_dictionary["epsilon"] = Categorical([0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                                                               0.6, 0.65, 0.7, 0.8, 0.9, 1, 2])
    hyperparameters_range_dictionary["zeta"] = Categorical(weight_list)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, itemKNNCF, recommenderELASTIC, recommenderCB, recommenderBetaGRAPH,
                                     recommederUserKNN, recommenderCYTHON],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )
    output_folder_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # RUN
    n_cases = 45
    metric_to_optimize = "MAP"
    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=30,
                           save_model="no",
                           output_folder_path=output_folder_path,
                           output_file_name_root=recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize=metric_to_optimize
                           )

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata")
    best_parameters = search_metadata["hyperparameters_best"]

    print("Best config for iteration %d is:" % i + "{}".format(best_parameters))

import time

from Notebooks_utils.data_splitter import train_test_holdout
from bayes_opt import BayesianOptimization

from Hybrid_of_predictions import ItemKNNScoresHybridRecommender
from Notebooks_utils.evaluation_function import evaluate_algorithm
import scipy.sparse as sps
from Hybrid_of_predictions import recommenderUserKNN
from Hybrid_of_predictions import recommenderCYTHON
from Hybrid_of_predictions import recommenderBetaGRAPH
from Hybrid_of_predictions import recommenderCB
from Hybrid_of_predictions import recommenderELASTIC
from Hybrid_of_predictions import itemKNNCF
from Hybrid_of_predictions import recommenderIALS
from mailer import mail_to


def run(alpha, beta, gamma, delta, epsilon, zeta, eta):
    urm_all = sps.load_npz('MyFiles/URM_all.npz')

    urm_train = sps.load_npz('URM/URM_train_for_tuning.npz')
    urm_test = sps.load_npz('URM/URM_test_for_tuning.npz')

    icm_all = sps.load_npz('MyFiles/ICM_all.npz')

    ###  SISTEMA I RECOMMENDER

    recommender = ItemKNNScoresHybridRecommender(urm_train, itemKNNCF, recommenderELASTIC, recommenderCB,
                                                 recommenderBetaGRAPH, recommenderUserKNN, recommenderCYTHON,
                                                 recommenderIALS)

    recommender.fit(alpha, beta, gamma, delta, epsilon, zeta, eta)

    return evaluate_algorithm(urm_test, recommender)["MAP"]


if __name__ == '__main__':
    # Bounded region of parameter space
    pbounds = {'alpha': (0, 5), 'beta': (0, 5), 'gamma': (0, 5), 'delta': (0, 5), 'epsilon': (0, 5),
               'zeta': (0, 5), 'eta': (0, 5)}

    optimizer = BayesianOptimization(
        f=run,
        pbounds=pbounds,
        verbose=2  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    start_time = time.time()

    optimizer.maximize(
        init_points=30,  # random steps
        n_iter=10,
    )

    end_time = time.time()

    print(optimizer.max)
    print("Elapsed time = {} minutes".format((end_time - start_time) / 60))

    elapsed_time = (end_time - start_time) / 60

    mail_to(toaddr="sommatab@gmail.com",
            subject="[NORMALISATION Tuning] Parameter search completed, this time with correct algorithms tunes",
            text="\nThe results are:\n" + str(optimizer.max) +
                 "\n Elapsed time: " + str(elapsed_time) + " mins" +
                 "\n Results obtained with current split split_number=" + 'URM/URM_train_for_tuning.npz' +
                 " on the parameterTuning repo.")

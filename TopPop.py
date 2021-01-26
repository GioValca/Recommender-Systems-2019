import zipfile
import scipy.sparse as sps
import numpy as np

from matplotlib import pyplot
from Algorithms import TopPopRecommender

dataFile = zipfile.ZipFile("/Users/giovanni/Desktop/RecSys/RecSysTest/Data/recommender-system-2019-challenge-polimi.zip")
URM_path = dataFile.extract("data_train.csv", path="/Users/giovanni/Desktop/RecSys/RecSysTest/Data")
URM_file = open(URM_path, 'r')

# for _ in range(10):
#   print(URM_file.readline())

# Start from beginning of the file
URM_file.seek(0)
numberInteractions = 0

for _ in URM_file:
    numberInteractions += 1

print("The number of interactions is {}".format(numberInteractions))

def rowSplit(rowString):
    split = rowString.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    result = tuple(split)

    return result


URM_file.seek(0)
URM_file.readline()
URM_tuples = []

for line in URM_file:
    URM_tuples.append(rowSplit(line))

userList, itemList, ratingList = zip(*URM_tuples)
userList = list(userList)
itemList = list(itemList)
ratingList = list(ratingList)

userList_unique = list(set(userList))
itemList_unique = list(set(itemList))

numUsers = len(userList_unique)
numItems = len(itemList_unique)

print("-- STATISTIC --")
print ("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
print ("Max ID items\t {}, Max Id users\t {}".format(max(itemList_unique), max(userList_unique)))
print ("Min items\t {}, Min user\t {}".format(min(itemList_unique), min(userList_unique)))
print ("Average interactions per user {:.2f}".format(numberInteractions/numUsers))
print ("Average interactions per item {:.2f}".format(numberInteractions/numItems))
print ("Sparsity {:.2f} %".format((1-float(numberInteractions)/(numItems*numUsers))*100))
print("---------------")

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all.tocsr()

#Item Activity

itemPopularity = (URM_all > 0).sum(axis=0)
itemPopularity = np.array(itemPopularity).squeeze()
itemPopularity = np.sort(itemPopularity)
pyplot.plot(itemPopularity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index')
pyplot.show()

tenPercent = int(numItems/10)
print("Average per-item interactions over the whole dataset {:.2f}".
      format(itemPopularity.mean()))
print("Average per-item interactions for the top 10% popular items {:.2f}".
      format(itemPopularity[-tenPercent].mean()))
print("Average per-item interactions for the least 10% popular items {:.2f}".
      format(itemPopularity[:tenPercent].mean()))
print("Average per-item interactions for the median 10% popular items {:.2f}".
      format(itemPopularity[int(numItems*0.45):int(numItems*0.55)].mean()))
print("Number of items with zero interactions {}".
      format(np.sum(itemPopularity==0)))

itemPopularityNonzero = itemPopularity[itemPopularity>0]
print("Non zero Statistic:")
tenPercent = int(len(itemPopularityNonzero)/10)
print("Average per-item interactions over the whole dataset {:.2f}".
      format(itemPopularityNonzero.mean()))
print("Average per-item interactions for the top 10% popular items {:.2f}".
      format(itemPopularityNonzero[-tenPercent].mean()))
print("Average per-item interactions for the least 10% popular items {:.2f}".
      format(itemPopularityNonzero[:tenPercent].mean()))
print("Average per-item interactions for the median 10% popular items {:.2f}".
      format(itemPopularityNonzero[int(numItems*0.45):int(numItems*0.55)].mean()))


pyplot.plot(itemPopularityNonzero, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index')
pyplot.show()

#User Activity

userActivity = (URM_all>0).sum(axis=1)
userActivity = np.array(userActivity).squeeze()
userActivity = np.sort(userActivity)
pyplot.plot(userActivity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('User Index')
pyplot.show()

#Splitting matrici
train_test_split = 0.80
numInteractions = URM_all.nnz
train_mask = np.random.choice([True,False], numInteractions, p=[train_test_split, 1-train_test_split])

URM_train = sps.load_npz("/Users/giovanni/Desktop/RecSys/RecSysTest/MyData/TrainAndTest/URM_train.npz")
URM_test = sps.load_npz("/Users/giovanni/Desktop/RecSys/RecSysTest/MyData/TrainAndTest/URM_test.npz")
URM_train = URM_train.tocsr()
URM_test = URM_test.tocsr()


topPopRecommender = TopPopRecommender()
topPopRecommender.fit(URM_train)
for user_id in userList_unique[0:10]:
    print(topPopRecommender.recommend(user_id, at=10))


#Evaluation metrics
#Precision
def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


#Recall
def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


#Mean Average Precision
def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


#Methods that call all theevaluation metrics
# We pass as paramether the recommender class
def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0
    for user_id in userList_unique:
        relevant_items = URM_test[user_id].indices
        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))


evaluate_algorithm(URM_test, topPopRecommender, at=10)
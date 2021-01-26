import pandas as pd
from tqdm import tqdm
from magics_with_UCM_region import choose_region


def print_to_csv_age(recommender1, recommender2, filename):
    target_file = pd.read_csv('dataset/data_target_users_test.csv')
    range_target_users = list(target_file["user_id"])
    cold_file = pd.read_csv('myFiles/coldUsers_id.csv')
    cold_users = list(cold_file["cold_users_id"])
    ages = pd.read_csv('dataset/data_UCM_age.csv')
    ages1 = ages.set_index("row", drop=False)
    users_with_age = list(ages["row"])

    output_file = open('output/'+filename+'.csv', mode='w')
    print('user_id,item_list', file=output_file)

    for user_id in tqdm(range_target_users):
        if user_id in cold_users:
            if user_id in users_with_age:
                age_of_user = ages1.loc[user_id, 'col']
                list_topPop_for_age = open('myFiles/TopPopForAge' + '{}'.format(age_of_user) + '.txt')
                items_rec = list_topPop_for_age.read().split('\n')
            else:
                items_rec = recommender2.recommend(user_id, at=10)
        else:
            items_rec = recommender1.recommend(user_id, at=10)
        sarr = [str(a) for a in items_rec]
        print('{},'.format(user_id) + ' '.join(sarr), file=output_file)

    output_file.close()


def print_to_csv_region(recommender1, recommender2, filename):
    target_file = pd.read_csv('dataset/data_target_users_test.csv')
    range_target_users = list(target_file["user_id"])
    cold_file = pd.read_csv('myFiles/coldUsers_id.csv')
    cold_users = list(cold_file["cold_users_id"])
    more_region_file = pd.read_csv('myFiles/users_with_more_regions.csv')
    more_region = list(more_region_file["user_id"])
    region = pd.read_csv('dataset/data_UCM_region.csv')
    region1 = region.groupby(['row', 'data'])['col'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    region2 = region1.set_index('row', drop=False)
    users_with_region = list(region["row"])

    #print(region2)
    cold_users_with_more_region = []

    for i in more_region:
        if i in cold_users:
            cold_users_with_more_region.append(i)


    output_file = open('output/'+filename+'.csv', mode='w')
    print('user_id,item_list', file=output_file)

    for user_id in tqdm(range_target_users):
        if user_id in cold_users:
            if user_id in users_with_region:
                if user_id in cold_users_with_more_region:
                    temp = region2.loc[user_id, 'col']
                    temp_list = [int(i) for i in temp if i != ',']
                    region_of_user = choose_region(temp_list)
                else:
                    region_of_user = region2.loc[user_id, 'col']

                list_topPop_for_region = open('myFiles/TopPopForRegion' + '{}'.format(region_of_user) + '.txt')
                items_rec = list_topPop_for_region.read().split('\n')
            else:
                items_rec = recommender2.recommend(user_id, at=10)
        else:
            items_rec = recommender1.recommend(user_id, at=10)
        sarr = [str(a) for a in items_rec]
        print('{},'.format(user_id) + ' '.join(sarr), file=output_file)

    output_file.close()


def print_to_csv_age_and_region(recommenderHybrid, recommenderTP, recommenderHybridSemiCold, recommenderUserCBF, filename):
    target_file = pd.read_csv('dataset/data_target_users_test.csv')
    range_target_users = list(target_file["user_id"])
    cold_file = pd.read_csv('myFiles/coldUsers_id.csv')
    cold_users = list(cold_file["cold_users_id"])
    semi_cold_file = pd.read_csv('myFiles/semiColdUsers_id.csv')
    semi_cold_users = list(semi_cold_file['semi_cold_users_id'])
    warm_user_file = pd.read_csv('myFiles/warm_users_id.csv')
    warm_users = list(warm_user_file['warm_users_id'])
    ages = pd.read_csv('dataset/data_UCM_age.csv')
    ages1 = ages.set_index("row", drop=False)
    users_with_age = list(ages["row"])
    both = pd.read_csv('region_user_combination/both.csv')
    both_list = list(both['user_with_both'])
    region = pd.read_csv('dataset/data_UCM_region.csv')
    region1 = region.groupby(['row', 'data'])['col'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    region2 = region1.set_index('row', drop=False)
    more_region_file = pd.read_csv('myFiles/users_with_more_regions.csv')
    more_region = list(more_region_file["user_id"])

    cold_users_with_more_region = []

    for i in more_region:
        if i in cold_users:
            cold_users_with_more_region.append(i)

    output_file = open('output/'+filename+'.csv', mode='w')
    print('user_id,item_list', file=output_file)

    for user_id in tqdm(range_target_users):

        if user_id in cold_users:
            '''
            most_similar_users = recommenderUserCBF.recommend(user_id)
            most_similar_user_without_cold = [x for x in most_similar_users if x not in cold_users]
            most_similar_user = most_similar_user_without_cold[0]
            items_rec = recommenderHybridSemiCold.recommend(most_similar_user)[0:10]
            '''

            if user_id in both_list:
                age_of_user = ages1.loc[user_id, 'col']
                if user_id in cold_users_with_more_region:
                    temp = region2.loc[user_id, 'col']
                    temp_list = [int(i) for i in temp if i != ',']
                    region_of_user = choose_region(temp_list)
                else:
                    region_of_user = region2.loc[user_id, 'col']
                list_topPop_for_age = open('myCombinations/TopPopForAge' + '{}'.format(age_of_user) + 'AndRegion{}'.format(region_of_user) + '.txt')
                items_rec = list_topPop_for_age.read().split('\n')
            else:
                if user_id in users_with_age:
                    age_of_user = ages1.loc[user_id, 'col']
                    list_topPop_for_age = open('myFiles/TopPopForAge' + '{}'.format(age_of_user) + '.txt')
                    items_rec = list_topPop_for_age.read().split('\n')
                else:
                    items_rec = recommenderTP.recommend(user_id)[0:10]

        if user_id in warm_users:
            items_rec = recommenderHybrid.recommend(user_id)[0:10]

        if user_id in semi_cold_users:
            items_rec = recommenderHybridSemiCold.recommend(user_id)[0:10]

        sarr = [str(a) for a in items_rec]
        print('{},'.format(user_id) + ' '.join(sarr), file=output_file)

    output_file.close()

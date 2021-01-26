from magics_with_UCM_region import get_users_with_region
from magics_with_UCM_ages import get_age_list
from tqdm import tqdm
import pandas as pd

user_with_region, list_user_divided_per_region = get_users_with_region()
age_list = get_age_list()

j = 1

both = open('region_user_combination/both.csv', mode='w')
print('user_with_both', file=both)

user_with_just_age = []
bothlist = []
print(age_list[3])
print(list_user_divided_per_region[6])
print(user_with_region)

# '''
for i in range(1, 11):
    for j in range(2, 8):
        string = 'region_user_combination/age' + str(i)
        string = string + 'region' + str(j) + '.csv'
        userXregionFile = open(string, mode='a')
        print('users', file=userXregionFile)
        userXregionFile.close()

j = 1

for age in age_list:
    for user in tqdm(age):
        if user in user_with_region:
            bothlist.append(user)
            string = 'region_user_combination/age' + str(j)
            x = 1
            for region in list_user_divided_per_region:
                if user in region:
                    string = string + 'region' + str(x) + '.csv'
                    userXregionFile = open(string, mode='a')
                    print('{}'.format(user), file=userXregionFile)
                    userXregionFile.close()
                x += 1
        else:
            user_with_just_age.append(user)
    j += 1

bothlist.sort()

for i in bothlist:
    print('{}'.format(i), file=both)
both.close()
# '''

# data = pd.read_csv('region_user_combination/both.csv')
# both_List = list(data["user_with_both"])


import pandas as pd

file = open('myFiles/NnzElementPerRow.txt', 'r')

interactions_per_user = file.readlines()

interactions_per_user = [x.strip() for x in interactions_per_user]

interactions_per_user = [int(x) for x in interactions_per_user]     #here i have the list of interactions per user as integers

#print(interactions_per_user)

'''
coldUsersDF = pd.DataFrame(columns=['warm_users_id'])

j = 0
for i in range(len(interactions_per_user)):
    if interactions_per_user[i] > 2:
        coldUsersDF.loc[j] = i
        j += 1

coldUsersDF.to_csv('myFiles/warm_users_id.csv', index=False)
'''

newfile = open("myFiles/cluster.csv", 'w')
print("interaction_class", file=newfile)
for i in interactions_per_user:
   if i==0 or i==1 or i==2:
      print("1", file=newfile)
   if i>2 and i<10:
      print("2", file=newfile)
   if i>=10:
      print("3", file=newfile)

import pickle
from collections import defaultdict
import random

# input_path = 'ml-1m.edges'
# with open(input_path, 'rb') as input_file:
#     data = pickle.load(input_file)
#
# output_path = 'LightGCN_data.txt'
# output_file = open(output_path, 'w')
#
# data = dict(sorted(data.items()))
#
# for userid in data:
#     line = ''
#     line += str(userid)
#     line += ' '
#     items = data[userid]
#     for i, itemid in enumerate(items):
#         line += str(itemid[0])
#         if i != len(items)-1:
#             line += ' '
#         else:
#             line += '\n'
#     output_file.write(line)

'''Generate user_list.txt item_list.txt test.txt train.txt'''
curr_userid = 0
curr_itemid = 0
remap_userid_dict = {}
remap_itemid_dict = {}
user_item_dict = defaultdict(list)

def write_to_file(output_file, user, items):
    global curr_userid
    global curr_itemid
    # Get remap user id
    if user not in remap_userid_dict:
        userid = curr_userid
        remap_userid_dict[user] = userid
        curr_userid += 1
    else:
        userid = remap_userid_dict[user]
    line = ''
    line += str(userid)
    line += ' '
    for i, item in enumerate(items):
        if item not in remap_itemid_dict:
            itemid = curr_itemid
            remap_itemid_dict[item] = itemid
            curr_itemid += 1
        else:
            itemid = remap_itemid_dict[item]
        line += str(itemid)
        if i != len(items)-1:
            line += ' '
        else:
            line += '\n'
    output_file.write(line)

# Get user_item_dict
fp = open("data/ratings.dat")
line = fp.readline()
while line:
    UserID, MovieID, Rating, _ = line.split("::")
    src = int(UserID)
    dest = int(MovieID)
    user_item_dict[src].append(dest)
    line = fp.readline()

# Write training data
output_train = 'train.txt'
output_train_file = open(output_train, 'w')

for user,items in user_item_dict.items():
    train_items = random.sample(items, k=round(len(items) * 0.8))
    user_item_dict[user] = list(set(items) - set(train_items))
    write_to_file(output_train_file, user, train_items)

# Write testing data
output_test = 'test.txt'
output_test_file = open(output_test, 'w')

for user,items in user_item_dict.items():
    write_to_file(output_test_file, user, items)

# Write remap user List
output_user_list = 'user_list.txt'
output_user_list_file = open(output_user_list, 'w')

output_user_list_file.write('org_id remap_id\n')
for org_id, remap_id in remap_userid_dict.items():
    line = str(org_id) + ' ' + str(remap_id) + '\n'
    output_user_list_file.write(line)

# Write remap item List
output_item_list = 'item_list.txt'
output_item_list_file = open(output_item_list, 'w')

output_item_list_file.write('org_id remap_id\n')
for org_id, remap_id in remap_itemid_dict.items():
    line = str(org_id) + ' ' + str(remap_id) + '\n'
    output_item_list_file.write(line)

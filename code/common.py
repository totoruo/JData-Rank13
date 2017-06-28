import pandas as pd
import pickle, os
from datetime import datetime, timedelta

action_1_path = "../data/JData_Action_201602.csv"
action_2_path = "../data/JData_Action_201603.csv"
action_3_path = "../data/JData_Action_201604.csv"
action_cate8_path = '../cache/actions_cate8.pkl'
comment_path = "../data/JData_Comment.csv"
product_path = "../data/JData_Product.csv"
user_path = "../data/JData_User.csv"

def extract_cate8():
    """
    筛选出cate=8的交互记录
    """
    action_1 = pd.read_csv(action_1_path)
    action_2 = pd.read_csv(action_2_path)
    action_3 = pd.read_csv(action_3_path)
    actions = pd.concat([action_1, action_2, action_3])
    actions = actions[actions['cate'] == 8]
    del actions['cate']
    pickle.dump(actions, open(action_cate8_path, 'wb'))

    return actions

def gen_action(start_date, end_date):
    """
    产生指定时间区间的行为数据
    """
    if os.path.exists(action_cate8_path):
        actions = pickle.load(open(action_cate8_path, 'rb'))
    else:
        actions = extract_cate8()
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

def gen_all_cate_action(start_date, end_date):
    """
    产生指定时间区间的all-cate行为数据
    """
    action_1 = pd.read_csv(action_1_path)
    action_2 = pd.read_csv(action_2_path)
    action_3 = pd.read_csv(action_3_path)
    actions = pd.concat([action_1, action_2, action_3])
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

def gen_labels(act_start_date, act_end_date, span=5):
    """
    产生交互日区间内的购买情况
    """
    act_end_date = datetime.strptime(act_start_date, '%Y-%m-%d') + timedelta(days=span)
    act_end_date = act_end_date.strftime('%Y-%m-%d')
    dump_path = '../cache/labels/labels_%s_%s.pkl' % (act_start_date, act_end_date)
    if os.path.exists(dump_path):
        labels = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(act_start_date, act_end_date)
        actions = actions[actions['type'] == 4]
        labels = actions[['user_id']].drop_duplicates()
        labels['label'] = 1
        print('buy user num is:', actions.shape[0])
        #pickle.dump(labels, open(dump_path, 'wb'))
    return labels

def gen_sample(end_date, span=5):
    """
    产生购买日前n天的交互用户
    """
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span)
    start_date = start_date.strftime('%Y-%m-%d')
    dump_path = '../cache/samples/samples_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        samples = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(start_date, end_date)
        samples = actions[['user_id']].drop_duplicates()
        print('samples num is:', samples.shape[0])
        #pickle.dump(samples, open(dump_path, 'wb'))
    return samples

def gen_truth(act_start_date, act_end_date):
    """
    产生交互日区间内的实际购买情况
    """
    dump_path = '../cache/labels/truth_%s_%s.pkl' % (act_start_date, act_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(act_start_date, act_end_date)
        actions = actions[actions['type'] == 4]
        actions = actions[['user_id']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def gen_submission(res, top):
    """
    产生最终提交的数据
    """
    res = res.sort_values(by='prob', ascending=False)
    res = res.head(top)
    res = res[['user_id']]
    res['sku_id'] = -1 #补商品为全-1
    res['user_id'] = res['user_id'].astype(int)
    res.to_csv('../cache/pred_user.csv', index=False, index_label=False)
    print('res num:', res.shape[0])
    return res

def user_score(index, pre_Y, truth, threshold):
    """
    计算F11得分
    """
    pred = gen_submission(index, pre_Y, threshold)
    truth = truth['user_id'].unique()
    pred = pred['user_id'].unique()
    pos, neg = 0,0
    for user_id in pred:
        if user_id in truth:
            pos += 1
        else:
            neg += 1
    if pos == 0:
        print(0)
        return
    print('hits', pos)
    Precise = 1.0 * pos / ( pos + neg)
    Recall = 1.0 * pos / len(truth)
    F11 = 6.0 * Precise * Recall / (5.0 * Recall + Precise)
    print('F11 score', F11)

if __name__ == '__main__':
    pass
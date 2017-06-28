import numpy as np
from dateutil.parser import parse
from common import *


def get_hours(start_date, end_date):
    d = parse(end_date) - parse(start_date)
    hours = int(d.days*24+d.seconds/3600)
    return hours

def gen_basic_user_feat():
    """
    用户基本特征
    """
    dump_path = '../cache/user_feature/basic_user_feat.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].replace({'-1':0,
                                           '15岁以下':1,
                                           '16-25岁':2,
                                           '26-35岁':3,
                                           '36-45岁':4,
                                           '46-55岁':5,
                                           '56岁以上':6,
                                           })
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        #pickle.dump(user, open(dump_path, 'wb'))
    return user

def gen_accumulate_user_feat(start_date, end_date):
    """
    用户累积特征
    """
    dump_path = '../cache/user_feature/acc_user_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(start_date, end_date)
        active_last = actions[['user_id', 'time']]
        buy_last = actions[['user_id', 'time', 'type']]
        active_days = actions[['user_id', 'time']]
        buy_days = actions[['user_id', 'time', 'type']]

        #活动天数
        active_days['time'] = active_days['time'].apply(lambda x: x[0:10])
        active_days = active_days.groupby(['user_id', 'time']).size().reset_index()
        active_days = active_days.groupby('user_id').size().reset_index()
        active_days.rename(columns={0: 'user_active_days'}, inplace=True)

        #购买天数
        buy_days = buy_days[buy_days['type'] == 4]
        del buy_days['type']
        buy_days['time'] = buy_days['time'].apply(lambda x: x[0:10])
        buy_days = buy_days.groupby(['user_id', 'time']).size().reset_index()
        buy_days = buy_days.groupby('user_id').size().reset_index()
        buy_days.rename(columns={0: 'user_buy_days'}, inplace=True)

        #最近交互时间(h)
        active_last = active_last.sort_values(by='time', ascending=False)
        active_last = active_last.drop_duplicates('user_id')
        active_last['user_active_last'] = active_last['time'].apply(lambda x: get_hours(x, end_date))
        del active_last['time']

        #最近购买时间(h)
        buy_last = buy_last[buy_last['type'] == 4]
        del buy_last['type']
        buy_last = buy_last.sort_values(by='time', ascending=False)
        buy_last = buy_last.drop_duplicates('user_id')
        buy_last['user_buy_last'] = buy_last['time'].apply(lambda x: get_hours(x, end_date))
        del buy_last['time']

        #四种统计量
        df = pd.get_dummies(actions['type'], prefix='user_action')
        actions_sku = pd.concat([actions[['user_id', 'sku_id']], df], axis=1)
        actions_sku = actions_sku.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions_sku['sku_id']
        actions_sku['user_action_sum'] = actions_sku['user_action_1'] + actions_sku['user_action_2'] + actions_sku['user_action_3'] + actions_sku['user_action_4'] + actions_sku['user_action_5'] + actions_sku['user_action_6']
        actions_mean = actions_sku.groupby(['user_id'], as_index=False).mean()
        actions_mean.rename(columns={'user_action_1': 'user_action_1_avg',
                                     'user_action_2': 'user_action_2_avg',
                                     'user_action_3': 'user_action_3_avg',
                                     'user_action_4': 'user_action_4_avg',
                                     'user_action_5': 'user_action_5_avg',
                                     'user_action_6': 'user_action_6_avg',
                                     'user_action_sum': 'user_action_sum_avg'}, inplace=True)
        actions_max = actions_sku.groupby(['user_id'], as_index=False).max()
        actions_max.rename(columns={'user_action_1': 'user_action_1_max',
                                    'user_action_2': 'user_action_2_max',
                                    'user_action_3': 'user_action_3_max',
                                    'user_action_4': 'user_action_4_max',
                                    'user_action_5': 'user_action_5_max',
                                    'user_action_6': 'user_action_6_max',
                                    'user_action_sum': 'user_action_sum_max'}, inplace=True)
        actions_min = actions_sku.groupby(['user_id'], as_index=False).min()
        actions_min.rename(columns={'user_action_1': 'user_action_1_min',
                                    'user_action_2': 'user_action_2_min',
                                    'user_action_3': 'user_action_3_min',
                                    'user_action_4': 'user_action_4_min',
                                    'user_action_5': 'user_action_5_min',
                                    'user_action_6': 'user_action_6_min',
                                    'user_action_sum': 'user_action_sum_min'}, inplace=True)
        actions_var = actions_sku.groupby(['user_id'], as_index=False).var()
        actions_var.rename(columns={'user_action_1': 'user_action_1_var',
                                    'user_action_2': 'user_action_2_var',
                                    'user_action_3': 'user_action_3_var',
                                    'user_action_4': 'user_action_4_var',
                                    'user_action_5': 'user_action_5_var',
                                    'user_action_6': 'user_action_6_var',
                                    'user_action_sum': 'user_action_sum_var'}, inplace=True)

        #交互统计和以及转化率
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['user_action_4'] / actions['user_action_1']
        actions['user_action_2_ratio'] = actions['user_action_4'] / actions['user_action_2']
        actions['user_action_5_ratio'] = actions['user_action_4'] / actions['user_action_5']
        actions['user_action_6_ratio'] = actions['user_action_4'] / actions['user_action_6']

        actions = pd.merge(actions, active_days, how='left', on='user_id')
        actions = pd.merge(actions, buy_days, how='left', on='user_id')
        actions = pd.merge(actions, active_last, how='left', on='user_id')
        actions = pd.merge(actions, buy_last, how='left', on='user_id')
        actions = pd.merge(actions, actions_mean, how='left', on='user_id')
        actions = pd.merge(actions, actions_max, how='left', on='user_id')
        actions = pd.merge(actions, actions_min, how='left', on='user_id')
        actions = pd.merge(actions, actions_var, how='left', on='user_id')

        #pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def gen_accumulate_user_feat_all_cate(start_date, end_date):
    """
    用户在全集上的累积特征
    """
    dump_path = '../cache/user_feature/acc_user_feat_all_cate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_all_cate_action(start_date, end_date)
        active_last = actions[['user_id', 'time']]
        active_days = actions[['user_id', 'time']]
        buy_days = actions[['user_id', 'time', 'type']]
        actions = actions[['user_id']].drop_duplicates()

        #活动天数
        active_days['time'] = active_days['time'].apply(lambda x: x[0:10])
        active_days = active_days.groupby(['user_id', 'time']).size().reset_index()
        active_days = active_days.groupby('user_id').size().reset_index()
        active_days.rename(columns={0: 'user_all_cate_active_days'}, inplace=True)

        #购买天数
        buy_days = buy_days[buy_days['type'] == 4]
        del buy_days['type']
        buy_days['time'] = buy_days['time'].apply(lambda x: x[0:10])
        buy_days = buy_days.groupby(['user_id', 'time']).size().reset_index()
        buy_days = buy_days.groupby('user_id').size().reset_index()
        buy_days.rename(columns={0: 'user_all_cate_buy_days'}, inplace=True)

        #最近交互时间(h)
        active_last = active_last.sort_values(by='time', ascending=False)
        active_last = active_last.drop_duplicates('user_id')
        active_last['user_all_cate_active_last'] = active_last['time'].apply(lambda x: get_hours(x, end_date))
        del active_last['time']

        actions = pd.merge(actions, active_days, how='left', on='user_id')
        actions = pd.merge(actions, buy_days, how='left', on='user_id')
        actions = pd.merge(actions, active_last, how='left', on='user_id')

        #pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def gen_user_action_feat(start_date, end_date, span):
    """
    指定区间内的用户商品交互特征(按天窗口）
    """
    dump_path = '../cache/user_feature/user_action_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(start_date, end_date)
        active_days = actions[['user_id', 'time']]
        product_nums = actions[['user_id', 'sku_id']]
        buy_days = actions[['user_id', 'time', 'type']]

        if span == 5 or span == 30:
            df = pd.get_dummies(actions['type'], prefix='%d_day_user_action' % (span))
            actions_sku = pd.concat([actions[['user_id', 'sku_id']], df], axis=1)
            actions_sku = actions_sku.groupby(['user_id', 'sku_id'], as_index=False).sum()
            del actions_sku['sku_id']
            actions_sku['%d_day_user_action_sum' % (span)] = actions_sku['%d_day_user_action_1' % (span)] + actions_sku['%d_day_user_action_2' % (span)] + actions_sku[
                '%d_day_user_action_3' % (span)] + actions_sku['%d_day_user_action_4' % (span)] + actions_sku['%d_day_user_action_5' % (span)] + actions_sku[
                                                 '%d_day_user_action_6' % (span)]
            actions_mean = actions_sku.groupby(['user_id'], as_index=False).mean()
            actions_mean.rename(columns={'%d_day_user_action_1' % (span): '%d_day_user_action_1_avg' % (span),
                                         '%d_day_user_action_2' % (span): '%d_day_user_action_2_avg' % (span),
                                         '%d_day_user_action_3' % (span): '%d_day_user_action_3_avg' % (span),
                                         '%d_day_user_action_4' % (span): '%d_day_user_action_4_avg' % (span),
                                         '%d_day_user_action_5' % (span): '%d_day_user_action_5_avg' % (span),
                                         '%d_day_user_action_6' % (span): '%d_day_user_action_6_avg' % (span),
                                         '%d_day_user_action_sum' % (span): '%d_day_user_action_sum_avg' % (span)}, inplace=True)
            actions_max = actions_sku.groupby(['user_id'], as_index=False).max()
            actions_max.rename(columns={'%d_day_user_action_1' % (span): '%d_day_user_action_1_max' % (span),
                                        '%d_day_user_action_2' % (span): '%d_day_user_action_2_max' % (span),
                                        '%d_day_user_action_3' % (span): '%d_day_user_action_3_max' % (span),
                                        '%d_day_user_action_4' % (span): '%d_day_user_action_4_max' % (span),
                                        '%d_day_user_action_5' % (span): '%d_day_user_action_5_max' % (span),
                                        '%d_day_user_action_6' % (span): '%d_day_user_action_6_max' % (span),
                                        '%d_day_user_action_sum' % (span): '%d_day_user_action_sum_max' % (span)}, inplace=True)
            actions_min = actions_sku.groupby(['user_id'], as_index=False).min()
            actions_min.rename(columns={'%d_day_user_action_1' % (span): '%d_day_user_action_1_min' % (span),
                                        '%d_day_user_action_2' % (span): '%d_day_user_action_2_min' % (span),
                                        '%d_day_user_action_3' % (span): '%d_day_user_action_3_min' % (span),
                                        '%d_day_user_action_4' % (span): '%d_day_user_action_4_min' % (span),
                                        '%d_day_user_action_5' % (span): '%d_day_user_action_5_min' % (span),
                                        '%d_day_user_action_6' % (span): '%d_day_user_action_6_min' % (span),
                                        '%d_day_user_action_sum' % (span): '%d_day_user_action_sum_min' % (span)}, inplace=True)
            actions_var = actions_sku.groupby(['user_id'], as_index=False).var()
            actions_var.rename(columns={'%d_day_user_action_1' % (span): '%d_day_user_action_1_var' % (span),
                                        '%d_day_user_action_2' % (span): '%d_day_user_action_2_var' % (span),
                                        '%d_day_user_action_3' % (span): '%d_day_user_action_3_var' % (span),
                                        '%d_day_user_action_4' % (span): '%d_day_user_action_4_var' % (span),
                                        '%d_day_user_action_5' % (span): '%d_day_user_action_5_var' % (span),
                                        '%d_day_user_action_6' % (span): '%d_day_user_action_6_var' % (span),
                                        '%d_day_user_action_sum' % (span): '%d_day_user_action_sum_var' % (span)}, inplace=True)


            # 交互总量
            df = pd.get_dummies(actions['type'], prefix='%d_day_user_action' % (span))
            actions = pd.concat([actions['user_id'], df], axis=1)  # type: pd.DataFrame
            actions = actions.groupby(['user_id'], as_index=False).sum()

            actions = pd.merge(actions, actions_mean, how='left', on='user_id')
            actions = pd.merge(actions, actions_max, how='left', on='user_id')
            actions = pd.merge(actions, actions_min, how='left', on='user_id')
            actions = pd.merge(actions, actions_var, how='left', on='user_id')
        else:
            # 交互总量
            df = pd.get_dummies(actions['type'], prefix='%d_day_user_action' % (span))
            actions = pd.concat([actions['user_id'], df], axis=1)  # type: pd.DataFrame
            actions = actions.groupby(['user_id'], as_index=False).sum()

        #做细最近一天特征
        if span == 1:
            #活动小时数
            active_days['time'] = active_days['time'].apply(lambda x: x[11:13])
            active_days = active_days.groupby(['user_id', 'time']).size().reset_index()
            active_days = active_days.groupby('user_id').size().reset_index()
            active_days.rename(columns={0: '1_day_user_active_hours'}, inplace=True)

            #交互商品种类
            product_nums = product_nums.groupby(['user_id', 'sku_id']).size().reset_index()
            product_nums = product_nums.groupby('user_id').size().reset_index()
            product_nums.rename(columns={0: '1_day_user_product_nums'}, inplace=True)

            actions = pd.merge(actions, active_days, how='left', on='user_id')
            actions = pd.merge(actions, product_nums, how='left', on='user_id')

            #pickle.dump(actions, open(dump_path, 'wb'))
            return actions

        #活动天数
        active_days['time'] = active_days['time'].apply(lambda x: x[0:10])
        active_days = active_days.groupby(['user_id', 'time']).size().reset_index()
        active_days = active_days.groupby('user_id').size().reset_index()
        active_days.rename(columns={0: '%d_day_user_active_days' % (span)}, inplace=True)

        #购买天数
        buy_days = buy_days[buy_days['type'] == 4]
        del buy_days['type']
        buy_days['time'] = buy_days['time'].apply(lambda x: x[0:10])
        buy_days = buy_days.groupby(['user_id', 'time']).size().reset_index()
        buy_days = buy_days.groupby('user_id').size().reset_index()
        buy_days.rename(columns={0: '%d_day_user_buy_days' % (span)}, inplace=True)

        actions = pd.merge(actions, active_days, how='left', on='user_id')
        actions = pd.merge(actions, buy_days, how='left', on='user_id')


        #pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def gen_user_action_feat_hour(start_date, end_date, span):
    """
    指定区间内的用户商品交互特征(按小时窗口）
    """
    dump_path = '../cache/user_feature/user_action_feat_%sh_%s.pkl' % (start_date.hour, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = gen_action(str(start_date), end_date)
        active_days = actions[['user_id', 'time']]

        #交互总量
        df = pd.get_dummies(actions['type'], prefix='%d_h_user_action' % (span))
        actions = pd.concat([actions['user_id'], df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).sum()

        #活跃小时数
        active_days['time'] = active_days['time'].apply(lambda x: x[11:13])
        active_days = active_days.groupby(['user_id', 'time']).size().reset_index()
        active_days = active_days.groupby('user_id').size().reset_index()
        active_days.rename(columns={0: '%d_h_user_active_hours' % (span)}, inplace=True)

        actions = pd.merge(actions, active_days, how='left', on='user_id')

        #pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def make_train_set(train_start_date, train_end_date, act_start_date, act_end_date, span=30):
    """
    构造训练集
    """
    dump_path = '../cache/dataset/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, act_start_date, act_end_date)
    if os.path.exists(dump_path):
        train_set = pickle.load(open(dump_path, 'rb'))
    else:
        train_set = gen_sample(train_end_date, span)
        label = gen_labels(act_start_date, act_end_date)

        basic_user_feat = gen_basic_user_feat()
        #删除无用的性别特征
        del basic_user_feat['sex_0.0']
        del basic_user_feat['sex_1.0']
        del basic_user_feat['sex_2.0']
        acc_user_feat = gen_accumulate_user_feat("2016-02-01", train_end_date)
        acc_user_feat_all_cate = gen_accumulate_user_feat_all_cate("2016-02-01", train_end_date)


        window = [1, 2, 3, 5, 7, 10, 15, 21, 30]
        for i in window:
            start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date = start_date.strftime('%Y-%m-%d')
            user_action_feat = gen_user_action_feat(start_date, train_end_date, i)
            train_set = pd.merge(train_set, user_action_feat, how='left', on='user_id')

        window = [4, 8, 16]
        for i in window:
            start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(hours=i)
            user_action_feat = gen_user_action_feat_hour(start_date, train_end_date, i)
            train_set = pd.merge(train_set, user_action_feat, how='left', on='user_id')

        train_set = pd.merge(train_set, basic_user_feat, how='left', on='user_id')
        train_set = pd.merge(train_set, acc_user_feat, how='left', on='user_id')
        train_set = pd.merge(train_set, acc_user_feat_all_cate, how='left', on='user_id')
        train_set = pd.merge(train_set, label, how='left', on='user_id')
        train_set = train_set.fillna({'label': 0})

        #pickle.dump(train_set, open(dump_path, 'wb'))


    label = train_set['label'].copy()
    del train_set['label']
    del train_set['user_id']
    return train_set, label

def make_test_set(test_start_date, test_end_date, span=30):
    """
    构造测试集
    """
    dump_path = '../cache/dataset/test_set_%s_%s.pkl' % (test_start_date, test_end_date)
    if os.path.exists(dump_path):
        test_set = pickle.load(open(dump_path, 'rb'))
    else:
        test_set = gen_sample(test_end_date, span)

        basic_user_feat = gen_basic_user_feat()
        del basic_user_feat['sex_0.0']
        del basic_user_feat['sex_1.0']
        del basic_user_feat['sex_2.0']
        acc_user_feat = gen_accumulate_user_feat("2016-02-01", test_end_date)
        acc_user_feat_all_cate = gen_accumulate_user_feat_all_cate("2016-02-01", test_end_date)

        window = [1, 2, 3, 5, 7, 10, 15, 21, 30]
        for i in window:
            start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date = start_date.strftime('%Y-%m-%d')
            user_action_feat = gen_user_action_feat(start_date, test_end_date, i)
            test_set = pd.merge(test_set, user_action_feat, how='left', on='user_id')

        window = [4, 8, 16]
        for i in window:
            start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(hours=i)
            user_action_feat = gen_user_action_feat_hour(start_date, test_end_date, i)
            test_set = pd.merge(test_set, user_action_feat, how='left', on='user_id')

        test_set = pd.merge(test_set, basic_user_feat, how='left', on='user_id')
        test_set = pd.merge(test_set, acc_user_feat, how='left', on='user_id')
        test_set = pd.merge(test_set, acc_user_feat_all_cate, how='left', on='user_id')


        #pickle.dump(test_set, open(dump_path, 'wb'))


    index = test_set[['user_id']].copy()
    del test_set['user_id']
    return index, test_set

if __name__ == '__main__':
    pass
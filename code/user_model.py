from xgboost import XGBClassifier
from collections import Counter
from gen_user_data import *

def offline_test():
    train_start_date = '2016-03-03'
    train_end_date = '2016-04-04'
    act_start_date = '2016-04-04'
    act_end_date = '2016-04-09'

    test_start_date = '2016-03-08'
    test_end_date = '2016-04-09'
    test_act_start_date = '2016-04-09'
    test_act_end_date = '2016-04-14'

    train_X, train_Y = make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
    # train_X, train_Y = make_train_set_slide(train_start_date, train_end_date, act_start_date, act_end_date)
    test_index, test_X = make_test_set(test_start_date, test_end_date)

    print('training...')
    c = Counter(train_Y.values)
    clf = XGBClassifier(max_depth=5, min_child_weight=6, scale_pos_weight=c[0] / 16 / c[1], nthread=12, seed=0)
    clf.fit(train_X.values, train_Y.values)

    pre_y = clf.predict_proba(test_X.values)[:,1]
    res = test_index.copy()
    res['prob'] = pre_y
    truth = gen_truth(test_act_start_date, test_act_end_date)
    pred = gen_submission(res, 710)
    user_score(pred, truth)

def online_submit():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    act_start_date = '2016-04-11'
    act_end_date = '2016-04-16'

    pre_start_date = '2016-03-15'
    pre_end_date = '2016-04-16'

    train_X, train_Y = make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
    pre_index, pre_X = make_test_set(pre_start_date, pre_end_date)

    print('training...')
    c = Counter(train_Y.values)
    clf = XGBClassifier(max_depth=3, min_child_weight=4, scale_pos_weight=c[0] / 16 / c[1], nthread=12, seed=0)
    clf.fit(train_X.values, train_Y.values)
    pre_y = clf.predict_proba(pre_X.values)[:,1]
    res = pre_index.copy()
    res['prob'] = pre_y
    pred_user = gen_submission(res, 1000)
    return pred_user

if __name__ == '__main__':
    online_submit()

# ============ 分类 ==============
    
def accuracy(y_true, y_pred):
    correct=0
    total=len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true==pred:
            correct+=1
    return correct/total

def precision(y_true, y_pred):
    true_positive=0
    false_positive=0

    for true, pred in zip(y_true, y_pred):
        if pred==1:
            if true==1:
                true_positive+=1
            else:
                false_positive+=1
    
    if true_positive+false_positive==0:
        return 0
    return true_positive/(true_positive+false_positive)

def recall(y_true, y_pred):
    true_positive=0
    false_negative=0
    for true, pred in zip(y_true, y_pred):
        if true==1:
            if pred==1:
                true_positive+=1
            else:
                false_negative+=1
    if true_positve+false_negative==0:
        return 0
    return true_positive/(true_positive+false_negative)


def get_roc_auc(y_true, y_pred):
    import numpy as np
    gt_pred=list(zip(y_true, y_pred))
    probs=[]
    pos_samples=[x for x in gt_pred if x[0]==1]
    neg_samples=[x for x in gt_pred if x[0]==0]

    for pos in pos_samples:
        for neg in neg_samples:
            if pos[1]>neg[1]:
                probs.append(1)
            elif pos[1]==neg[1]:
                probs.append(0.5)
            else:
                probs.append(0)
    return np.mean(probs)


def compute_gauc(y_true, y_pred, group_ids):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    unique_groups=np.unique(group_ids)

    auc_values=[]
    for group in unique_groups:
        group_mask=group_ids==group
        group_y_true=y_true[group_mask]
        group_y_pred=y_pred[group_mask]

        # 计算该组的auc值
        group_auc=roc_auc_score(group_y_true, group_y_pred)
        auc_values.append(group_auc)

    # 计算auc的mean值
    gauc=np.mean(auc_values)
    return gauc


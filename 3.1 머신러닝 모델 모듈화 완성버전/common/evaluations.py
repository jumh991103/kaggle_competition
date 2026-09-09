#modeling파일에 fit_evaluation함수 만들다 보니까 evaluation관련된걸 modling파일에 만들기엔 좀 맘에 안들어서 이 파일 만들었음.
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

def get_auc_score(y,pred):
    fpr, tpr, _ =roc_curve(y,pred)
    return auc(fpr, tpr)

def get_f1_score(y, pred, average='weighted'):
    return f1_score(y, pred, average=average)
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from .modeling import Modeling, CAT_COLS   # common 패키지 안에 둘 경우 상대경로 import


#교차 검증용 함수
#input parameter가 뭐가 필요할까?
# (필수) ->  model, train
# (옵션) -> target_col_nameㅏ, n_splits, shuffle, random_state
# 필수와 옵션은 어떻게 나눈거야? 왜 필수인지 왜 옵션인지? 옵션은 default값을 정의 할수있는것, 필수는 default값을 정의 할수없는것
# ?? target_col_name을 어떻게 정의함? 정의할순 없지만 non이라고는 할 수 잇음. 왜냐면 회귀모델일수도 있으니까, 회귀모델일땐 컬럼이름이 필요없음
# 언제 Kfold를 쓸찌 언제 Stratified를 쓸지 어떻게 구분할까? 파라미터를 추가해야할까? target컬럼 명을 가지고 none이면 회귀 아니면 분류인거니까 ㅇㅇ(주석을 달아서 이걸 설명해놔야함.쓰는 이유는 우린 ai를 통해 더 발전시킬거니까 이런걸 적어둬야함.)
#만들다 보니까 eval_fnc라는 함수가 필요하고 default를 accuracy로 넣어도 큰상관없으니까 옵션으로 뒀음. 그리고 target이름은 분류든 회귀든 필수라 필수로 넣고 대신 분류 회귀 나눠주기 위해 is_classifier를 추가했음.

def do_cross_validation(
        model:object, train:pd.DataFrame, target_col_name:str,  #object는 뭐지 뭔지 알수없을때 쓰는건가.

        is_classifier:bool=False, #회귀냐 분류나 True면 분류임.
        use_proba: bool = False,          # True면 predict_proba()[:, 1] 사용 (AUC용)
        eval_fnc:object = accuracy_score,
        n_splits:int = 5,
        shuffle:bool = True,
        random_state:int = 42,
        **kargs #추가적으로 더 있을 수 있으니.
):
    """
    Cross validation 객체 생성
    만약 is classifier가 True면 분류 모델이고, 그렇지 않으면 회귀 모델임.
    """
    if is_classifier:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state, **kargs)

        gen_cv = skf.split(train, train[target_col_name]) #generator가 리턴되니까 gen이라 쓴거임.
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state, **kargs)

        gen_cv = kf.split(train)

    """
    Cross validation을 이용한 모델 학습
    """

    lst_scores = []

    #학습 시작

    for tr_index, val_index in gen_cv:

        #학습용, 검증용 데이터 생성
        x_tr, y_tr=train.iloc[tr_index].drop([target_col_name], axis=1), train.iloc[tr_index][target_col_name]
        x_te, y_te=train.iloc[val_index].drop([target_col_name], axis=1), train.iloc[val_index][target_col_name]

        #모델 학습
        model.fit(x_tr, y_tr)
        #예측
        if use_proba:
            pred = model.predict_proba(x_te)[:, 1]
        else:
            pred = model.predict(x_te)
        #평가
        lst_scores.append(eval_fnc(y_te, pred))

    #학습 완료
    print(f"교차 검증 점수({eval_fnc.__name__}) : {np.mean(lst_scores)}")
    return lst_scores


# ↓↓↓ 여기부터 새로 추가 — Modeling(xgb/lgb/cat 자동선택) 전용 CV 함수 ↓↓↓
def do_cross_validation_with_modeling(
        train: pd.DataFrame, target_col_name: str,
        cat_cols: list = CAT_COLS,
        is_classifier: bool = True,   # 이 대회는 항상 분류라 기본값 True로 둠
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        **kargs
):
    """
    Modeling 클래스는 model.fit()/model.predict() 형태가 아니라
    생성자에서 바로 x_tr/y_tr을 받고 fit_evaluation()으로 3개 모델을 한번에 학습+평가하는 구조라
    do_cross_validation()에 그대로 못 끼워넣음 -> 전용 버전을 따로 만듦.
    폴드마다 데이터가 다르므로 Modeling 객체도 폴드마다 새로 생성해야 함.
    """
    if is_classifier:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state, **kargs)
        gen_cv = skf.split(train, train[target_col_name])
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state, **kargs)
        gen_cv = kf.split(train)

    lst_scores = []
    lst_train_scores = []
    lst_best_models = []
    n_iter = 0

    for tr_index, val_index in gen_cv:
        n_iter += 1

        x_tr, y_tr = train.iloc[tr_index].drop([target_col_name], axis=1), train.iloc[tr_index][target_col_name]
        x_te, y_te = train.iloc[val_index].drop([target_col_name], axis=1), train.iloc[val_index][target_col_name]

        modeling = Modeling(x_tr=x_tr, y_tr=y_tr, cat_cols=cat_cols)
        modeling.fit_evaluation(x_te=x_te, y_te=y_te)

        best = modeling.get_best_model()
        lst_train_scores.append(best['train_score'])
        lst_scores.append(best['test_score'])      # Modeling이 이미 AUC(get_auc_score)로 계산해둔 값
        lst_best_models.append(best['model_name'])

        print(f"{n_iter}번째 Fold — 선택된 모델: {best['model_name']}, "
              f"train AUC: {best['train_score']:.4f}, valid AUC: {best['test_score']:.4f}")

    print('-' * 50)
    print(f"교차 검증 valid AUC 평균: {np.mean(lst_scores):.4f} (std: {np.std(lst_scores):.4f})")
    print(f"폴드별 선택된 모델: {lst_best_models}")

    return lst_scores, lst_train_scores, lst_best_models
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score


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
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state, **kargs)

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
        x_te, y_te=train.iloc[val_index].drop([target_col_name], axis=1), train.iloc[val_index][target_col_name] # 여길 하다보니 targetname은 회귀든 분류든 무조건 필요한걸 알게되어서 위에 수정할거임. 그리고 왜 iloc쓴건지 모르겠네..? 갑자기 예제코드에서 오류나셔서 고치시더니 이걸 붙이셨음.

        #모델 학습
        model.fit(x_tr, y_tr)
        #예측
        pred = model.predict(x_te)
        #평가
        lst_scores.append(eval_fnc(y_te, pred)) #평가해주는 함수 조금있다가 새로 만들거임. 평가가 n개가 만들어질거아냐? 그걸 리스트에 담을거임.

    #학습 완료
    print(f"교차 검증 점수({eval_fnc.__name__}) : {np.mean(lst_scores)}") #eva_fnc.__name__ :  eval_fnc는 함수니까 이렇게 작성하면 함수이름이 나옴
    return lst_scores #이건 필요에 따라 넘겨도 되고 안넘겨도 됨.
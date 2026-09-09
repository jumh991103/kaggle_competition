import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import enum 
# plot_importance라는 함수 이름이 똑같잖아. 그럼 만약 강의자료에서 쓴대로 쓰면 위에는 사라지고 아래꺼만 살아남음. 그래서 이름을 따로 부르도록했음.
from tqdm.auto import tqdm

from xgboost import XGBClassifier
from xgboost import plot_importance as xgb_plot_importance
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as lgb_plot_importance
from catboost import CatBoostClassifier

from .utils import reset_seeds
from .evaluations import get_auc_score


#catboost는 importance가 없어서 함수로 만들었는데 이것도 이름(명명 규칙)을 나머지 모델과 같이 썼음.
def cat_plot_importance(cat):
    feature_importance = cat.feature_importances_
    feature_names = np.array(cat.feature_names_)
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(12, 6))
    plt.barh(
        range(len(sorted_idx)),
        feature_importance[sorted_idx]
    )
    plt.yticks(
        range(len(sorted_idx)),
        feature_names[sorted_idx]
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.show()

class BoostModelType(enum.Enum): # (enum랜덤번호, 모델, 하이퍼파라미터, importance함수)
    xgb = (enum.auto(), XGBClassifier, {'tree_method':'hist', 'enable_categorical':True}, xgb_plot_importance)
    lgb = (enum.auto(), LGBMClassifier, {'verbose': -1}, lgb_plot_importance)
    cat = (enum.auto(), CatBoostClassifier, {'verbose':0}, cat_plot_importance)

    @classmethod
    def show_plot_importance(cls, model):
        for model_type in BoostModelType:
            if isinstance(model, model_type.value[1]):
                model_type.value[3](model)
                plt.show()


class Modeling: #학습 진행하는 클래스 세팅
    #init안에서 선언하는 변수들은 method안에서 쓸 변수들 선언하는거임.
    def __init__(
        self, x_tr:pd.DataFrame, y_tr:pd.DataFrame, #학습데이터
        vali_func=get_auc_score,  #평가함수
        cat_cols = [ 
            'pclass', 'sex', 'embarked', 'who', 'adult_male', 'deck', 'alone'
        ]) -> None:
        self.__best_model = { #얘는 외부에서 봐야하지만 조작은 못하게 하기위해서. 대신 외부에서 봐야하니까 아래 get_best_model이라는 함수로 외부에서 볼수 있도록 함.
    
            'model_name': None,
            'hpo': None,
            'train_score': 0.0,
            'test_score': 0.0,
            'score_type': None
        }
        self.__vali_func = vali_func
        self.__cat_cols = cat_cols
        self.__targets = y_tr #feature와 target은 프라이빗하게
        self.__features = x_tr 
        self.__convert_dtype(self.__features)
        self.__valid_features_targets() #이걸 왜 init함수 안에 넣었찌? init함수에는 변수만 넣는거 아냐? ->학습하기전에 feature와 target을 만들어냄으로써 만약 학습할 데이터 자체가아니면 굳이 생성조차 할 필요가 없으므로


    #사용불가능한 데이터타입들을 가능하게 변환하는건 test도 해야함. 똑같은 코드 두번써야함.->재사용해야하니까 함수로 만들자.
    def __convert_dtype(self, features):
        bool_cols = features.select_dtypes(include=["bool", "boolean"]).columns
        features[bool_cols] = features[bool_cols].astype("int8")
        
        features[self.__cat_cols] = features[self.__cat_cols].astype('category')


    def __valid_features_targets(self,x_te:pd.DataFrame=None, y_te:pd.DataFrame=None): #feature와 타겟 검증
        assert self.__features.isnull().sum().sum() == 0, "features에 결측치가 있습니다."
        assert len(self.__features) == len(self.__targets), "features와 targets의 데이터 수가 다릅니다."

        if x_te is not None and y_te is not None:
            assert x_te.isnull().sum().sum() == 0, "[평가용]features에 결측치가 있습니다."
            assert len(self.__features) == len(self.__targets), "[평가용] features와 targets의 데이터 수가 다릅니다."
            assert self.__features.shape[1] ==x_te.shape[1], "[평가용] features 수가 다릅니다."
        
    def get_best_model(self):
        return self.__best_model #프라이빗 변수이기떄문에 이 함수를 통해서 return받은거임.

    def __fit(self, model_type:BoostModelType, add_hpo:dict):  #원래 dict={} 이거였는데 왜 dict로 바꿈?

        assert model_type in BoostModelType, "정상적인 모델타입이 아닙니다"
        
        #하이퍼 파라미터 정의
        hpo = model_type.value[2] | add_hpo #boostmodeltype 클래스에서 써놓은 기본 하이퍼파라미터랑 우리가 조작해서 새로 넣을 하이퍼파라미터를 합쳐서 한번에 사용해야하니까
        if model_type is BoostModelType.cat:
            hpo = hpo | {'cat_features' : self.__cat_cols}
        #모델 생성
        model = model_type.value[1](**hpo)
        #모델 학습
        model.fit(self.__features, self.__targets) 

        return model, hpo

    def __evaluation(self, model, hpo, y_te, x_te):
        #모델 평가
        test_score = self.__vali_func(y=y_te, pred=model.predict(x_te))

        if self.__best_model['test_score'] < test_score:
            self.__best_model = {
                        'model': model,
                        'model_name': model.__class__.__name__, #클래스명이 리턴됨.
                        'hpo': hpo,
                        'train_score': self.__vali_func(y=self.__targets, pred=model.predict(self.__features)),
                        'y_pre_te' : model.predict(x_te),
                        'test_score': test_score,
                        'score_type': 'auc' #얘는 나중에 바꿔주신다고함.
                    }

    @reset_seeds()
    def fit_evaluation(self, y_te:pd.DataFrame, x_te:pd.DataFrame, add_hpo:dict={})-> None:
        self.__convert_dtype(x_te)
        #세개의 모델을 각각 학습 및 평가해야하니까 for문 돌려야겠지?
        for model_type in tqdm(BoostModelType, desc="training.."): #진행바
            try:
                model, hpo = self.__fit(model_type, add_hpo)
                self.__evaluation(model, hpo, y_te, x_te)
            except:
                print(f"오류발생: {model_type.value[1].__class__.__name__}")
    def predict_by_best_model(self, features):
        return self.__best_model['model'].predict(features)



    # @reset_seeds()
    # def fit_evaluation(self, model_type:BoostModelType, add_hpo:dict={}): #학습 + 평가까지 하는 함수 ->기능이 두개인 함수네..?그래서 이걸 쪼개야함.(쪼개기전에 fit_evaluation함수였음.)

    #     assert model_type in BoostModelType, "정상적인 모델타입이 아닙니다"

    #     #하이퍼 파라미터 정의
    #     hpo = model_type.value[2] | add_hpo #boostmodeltype 클래스에서 써놓은 기본 하이퍼파라미터랑 우리가 조작해서 새로 넣을 하이퍼파라미터를 합쳐서 한번에 사용해야하니까
    #     #모델 생성
    #     model = model_type.value[1]
    #     #모델 학습
    #     model.fit(self.__features, self.__targets) #이 부분 코드가 잘 이해가 안된다. 학습시킨걸 리턴시키는거지?
    #     #모델 평가
    #     self.__best_model['train_score'] = self.__vali_func(y=self.__targets, pred=model.predict(self.__features)) #이건 학습데이터 평가잖아? 우리가 원하는건 테스트 데이터가 들어가는거임.(나중에 오버피팅 해석하려면 얘도 필요하긴해서 저장하긴 해야함.그래서 init함수에 train_score라고 저장할 곳을 만들어놨음)

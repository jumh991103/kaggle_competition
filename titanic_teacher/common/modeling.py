import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import enum 
from tqdm.auto import tqdm

from xgboost import XGBClassifier
from xgboost import plot_importance as xgb_plot_importance
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as lgb_plot_importance
from catboost import CatBoostClassifier

from .utils import reset_seeds
from .evaluations import get_auc_score

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


class BoostModelType(enum.Enum):
    xgb = (enum.auto(), XGBClassifier, 
            {'tree_method':'hist', 'enable_categorical':True}, xgb_plot_importance)
    lgb = (enum.auto(), LGBMClassifier, 
            {'verbose': -1}, lgb_plot_importance)
    cat = (enum.auto(), CatBoostClassifier, 
            {'verbose':0}, cat_plot_importance)

    @classmethod
    def show_plot_importance(cls, model):
        for model_type in BoostModelType:
            if isinstance(model, model_type.value[1]):
                model_type.value[3](model)
                plt.show()



class Modeling:

    def __init__(
        self, x_tr:pd.DataFrame, y_tr:pd.DataFrame, 
        vali_func=get_auc_score, 
        cat_cols = [
            'pclass', 'sex', 'embarked', 'who', 'adult_male', 'deck', 'alone'
        ]) -> None:
        self.__best_model = {
            'model_name': None,
            'hpo': None,
            'train_score': 0.0,
            'test_score': 0.0,
            'score_type': None
        }
        self.__vali_func = vali_func
        self.__cat_cols = cat_cols
        self.__targets = y_tr
        self.__features = x_tr 
        self.__valid_features_targets()
        self.__convert_dtype(self.__features)

    def __convert_dtype(self, features):
        bool_cols = features.select_dtypes(include=["bool", "boolean"]).columns
        features[bool_cols] = features[bool_cols].astype("int8")

        features[self.__cat_cols] = features[self.__cat_cols].astype('category')

    def __valid_features_targets(self, x_te:pd.DataFrame=None, y_te:pd.DataFrame=None) -> None:
        assert self.__features.isnull().sum().sum() == 0, "[학습용] features에 결측치가 있습니다."
        assert len(self.__features) == len(self.__targets), "[학습용] features와 targets의 데이터 수가 다릅니다."

        if x_te is not None and y_te is not None:
            assert x_te.isnull().sum().sum() == 0, "[평가용] features에 결측치가 있습니다."
            assert len(self.__features) == len(self.__targets), "[평가용] features와 targets의 데이터 수가 다릅니다."
            assert self.__features.shape[1] == x_te.shape[1], "[평가용] features 수가 다릅니다."



    def get_best_model(self):
        return self.__best_model

    def __fit(self, model_type:BoostModelType, add_hpo:dict):

        assert model_type in BoostModelType, "정상적인 모델 타입이 아닙니다."

        # 하이퍼 파라미터 정의 
        hpo = model_type.value[2] | add_hpo
        if model_type is BoostModelType.cat:
            hpo = hpo | {'cat_features': self.__cat_cols}

        # 모델 생성 
        model = model_type.value[1](**hpo)

        # 모델 학습 
        model.fit(self.__features, self.__targets) 

        return model, hpo 

    def __evaluation(self, model, hpo, y_te, x_te):
        # 모델 평가 
        test_score = self.__vali_func(y=y_te, pred=model.predict(x_te))

        if self.__best_model['test_score'] < test_score:
            self.__best_model = {
                'model':model,
                'model_name': model.__class__.__name__,
                'hpo': hpo,
                'train_score': self.__vali_func(y=self.__targets, pred=model.predict(self.__features)),
                'test_score': test_score,
                'score_type': self.__vali_func.__name__
            }


    @reset_seeds()
    def fit_evaluation(
        self, y_te:pd.DataFrame, x_te:pd.DataFrame, add_hpo:dict={}) -> None:
        
        self.__valid_features_targets(x_te, y_te)
        self.__convert_dtype(x_te)

        for model_type in tqdm(
            list(BoostModelType.__members__), desc="training.."):

            try:
                model, hpo = self.__fit(BoostModelType[model_type], add_hpo)
                self.__evaluation(model, hpo, y_te, x_te)
            except:
                print(f"오류 발생: {BoostModelType[model_type].value[1].__class__.__name__}")
        
    def predict_by_best_model(self, features):
        return self.__best_model['model'].predict(features)



import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split 

def reset_seeds(seed=42):       # 1. 데코레이터 파라미터 받기
    def decorator(func):        # 2. 데코레이션 대상 함수 받기
        def wrapper_func(*args, **kwargs):  # 3. 함수 실행 시 인자 받기
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
#위 세줄로 랜덤시드를 고정시키면 다른 곳에서 따로 고정안시켜도됨.대신 모든 함수에 적용되어야하니까 이떄 쓰는게 데코레이션임.
            return func(*args, **kwargs)

        return wrapper_func

    return decorator

#이 아래는 데코레이터 함수 보여주기 위해 한번 적어보신건가? 클로드야 알려줘.

@reset_seeds() #seed바꾸고 싶으면 괄호안에 seed= x 이런식으로 넣으면 시드 바꿀수있음.
def train_test_split_by_target(
    df:pd.DataFrame, target_name:str='survived') -> tuple:

    return train_test_split(
        df, stratify=df[target_name]
    )
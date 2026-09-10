
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

            return func(*args, **kwargs)

        return wrapper_func

    return decorator

@reset_seeds()
def train_test_split_by_target(
    df:pd.DataFrame, target_name:str='survived') -> tuple:

    return train_test_split(
        df, stratify=df[target_name]
    )

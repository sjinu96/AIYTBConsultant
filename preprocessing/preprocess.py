import os
from datetime import datetime, date

import pandas as pd
import numpy as np

from sklearn import utils
from sklearn import preprocessing



class Preprocessor:
    """raw 데이터를 입력 받아 전처리를 진행하는 함수
    변수를 추가하는 함수는 크게 '_get_xxxx'와 '_add_xxxx'로 구성
    """

    def _fit_transform(self, raw):
        result = raw.copy()

        result = self._n_comment_to_float(result)
        result = self._str_to_datetype(result)
        result = self._add_n_hashtag(result)

        non_numeric = ["channel", "title", "genre", "description", "date", "sign_in"]

        result = self._merge(result, non_numeric)
        result = self._add_sub_diff(result)
        result = self._add_no_upload_interval(result)
        result = self._remove_nan(result, non_numeric)

        return result

    '''add and modify features'''
    def _n_comment_to_float(self, result):
        idx1 = result["n_comment"] == "댓글 사용 중지"
        idx2 = result.n_comment.isna()
        idx = idx1 | idx2
        result["n_comment"].loc[idx] = result["n_comment"].loc[idx].apply(lambda x: 0)
        result["n_comment"] = result["n_comment"].astype(float)
        return result


    def _str_to_datetype(self, result):
        """csv파일 로드시 date 컬럼이 str 타입으로 읽혀진 경우 이를 datetype으로 변환"""
        if pd.api.types.is_datetime64_ns_dtype(result["date"]):
            pass
        else:
            result["date"] = pd.to_datetime(result["date"])
        return result


    def _add_n_hashtag(self, result):
        result["n_hashtage"] = 0
        idx = result["description"].notnull()
        result.loc[idx, "n_hashtage"] = result.loc[idx, "description"].apply(
            lambda x: len(x.split("#")) - 1
        )
        return result


    def _merge(self, result, non_numeric):
        # operates both merge and creating video_num featrue
        numeric = [col for col in result.columns.tolist() if col not in non_numeric]
        return (
            result.groupby(["channel", "date"])
            .apply(lambda x: self._get_to_merge(x, numeric, non_numeric))
            .reset_index(drop=True)
        )
    

    def _add_sub_diff(self, result):
        result = (
            result.groupby("channel")
            .apply(lambda x: self._get_sub_diff(x))
            .reset_index(drop=True)
        )
        result.loc[result.sub_diff.isna(), "sub_diff"] = 0
        return result
    

    def _add_no_upload_interval(self, result):
        return (
            result.groupby("channel")
            .apply(lambda x: self._get_no_upload_interval(x))
            .reset_index(drop=True)
        )


    @staticmethod
    def _get_to_merge(data, numeric, non_numeric):
        data.reset_index(drop=True)
        num_to_add = data.title.shape[0] - data.title.isna().sum()
        data = pd.concat((data[non_numeric].iloc[0, :], data[numeric].mean()))
        data["video_num"] = num_to_add
        return data


    @staticmethod
    def _get_sub_diff(result):
        '''하루 단위 구독자 변화량 변수를 추가하는 함수'''
        result = result.reset_index(drop=True)
        result["sub_diff"] = result["cumul_subs"] - result["cumul_subs"].shift()
        return result


    @staticmethod
    def _get_no_upload_interval(result):
        result = result.reset_index(drop=True)
        upload_idx = result[result["video_num"] != 0].index.tolist()
        temp = [0 for i in range(result.shape[0])]
        for i in range(len(upload_idx)):
            if i == len(upload_idx) - 1:
                former = upload_idx[i]
                temp[former + 1 :] = [i + 1 for i in range(len(temp[former + 1 :]))]
            else:
                former, latter = upload_idx[i], upload_idx[i + 1]
                temp[former + 1 : latter] = [
                    i + 1 for i in range(len(temp[former + 1 : latter]))
                ]
        result["no_upload_interval"] = temp
        return result
    

    def _remove_nan(self, result, non_numeric):
        numeric = [col for col in result.columns.tolist() if col not in non_numeric]
        result.loc[:, numeric] = result.loc[:, numeric].fillna(0)
        return result

    '''create sequential data'''
    def _extract_at_least_filter(self, result, filter_size):
        '''fillter_size 이상인 채널을 추출하는 함수'''
        alive_idx = result["channel"].value_counts() >= filter_size
        alive_array = alive_idx[alive_idx == True].index
        return result[result["channel"].isin(alive_array)].reset_index(drop=True)


    def _create_sequential_data(
        self,
        result,
        filter_size=7,
        target_size=1,
        stride=1,
        drop_features=None,
        target_features=None,
    ):
    '''sequential 데이터를 생성하는 함수
    1. filter_size와 target_size를 고려했을 때 정보량이 너무 적은 채널은 삭제함
    2. 학습에 필요 없는 컬럼은 제거
    '''
        
        result = self._extract_at_least_filter(result, filter_size + target_size)

        # drop_features: features to drop fromf X (features)
        # target_features: features to extract from Y (targets)
        if drop_features is None:
            drop_features = [
                "date",
                "genre",
                "title",
                "channel",
                "description",
                "sign_in",
                "current_cumul_view",
                "current_n_video",
                "current_cumul_subs",
            ]
        if target_features is None:
            target_features = ["sub_diff"]

        # return train, target set wrt groups
        result = (
            result.groupby("channel")
            .apply(
                lambda x: self._to_sequential(
                    x, filter_size, target_size, stride, drop_features, target_features
                )
            )
            .reset_index(drop=True)
        )
        return self._combine(result)
    

    @staticmethod
    def _to_sequential(
        result, filter_size, target_size, stride, drop_features, target_features
    ):
        result = result.reset_index(drop=True)
        idx_list = result.index.tolist()

        train, target = [], []
        for i in range((len(idx_list) - filter_size - target_size) // stride + 1):
            train_idx = idx_list[i * stride : i * stride + filter_size]
            target_idx = idx_list[
                i * stride + filter_size : i * stride + filter_size + target_size
            ]
            train_temp = result.loc[train_idx, :].values.reshape(1, -1)
            target_temp = result.loc[target_idx, target_features].values.reshape(1, -1)

            train = train_temp.copy() if i == 0 else np.vstack([train, train_temp])
            target = target_temp.copy() if i == 0 else np.vstack([target, target_temp])

        train = pd.DataFrame(train, columns=result.columns.tolist() * filter_size)
        target = pd.DataFrame(target, columns=target_features * target_size)
        return train.drop(drop_features, axis=1), target


    def _combine(self, result):
        temp0, temp1 = [], []
        for i in range(len(result)):
            temp0.append(result[i][0])
            temp1.append(result[i][1])
        temp0 = pd.concat(temp0)
        temp1 = pd.concat(temp1)
        return (temp0, temp1)

    '''scaling '''
    def scale(self, data, return_original_scale=True):
        original_scale = pd.concat((data.max(), data.min()), axis=1).T
        original_scale.index = ["max", "min"]
        scaler = preprocessing.MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        if return_original_scale:
            return data, original_scale
        return data


    def inverse_scale(pred, scl):
        for idx in range(pred.shape[1]):
            minimum = scl.iloc[1, idx]
            maximum = scl.iloc[0, idx]
            original = pred.iloc[:, idx]
            pred.iloc[:, idx] = (maximum - minimum) * original + minimum

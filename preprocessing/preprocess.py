import os
from datetime import datetime, date

import pandas as pd
import numpy as np

'''
Preprocessor
get input data and apply entire preprocessing for train
'''


class Preprocessor:
    IS_TRANSFORM = False
    def fit_transform(self, data_path, return_output=True):
        self.raw = pd.read_csv(data_path)
        self.result = self.raw.copy()

        self.str_to_datetype()
        self.add_is_upload()
        self.add_sub_diff()
        self.add_no_upload_interval()
        self.add_n_hashtag()
        self.IS_TRANSFORM = True
        
        if return_output:
            return self.result
        else:
            print('preprocess finished.')
    
    def get_train_data(self, filter_size: 'time interval'=7, drop_features=None):
        if self.IS_TRANSFORM!=True:
            raise NotImplementedError("You need to run 'fit_transform' at first.")
        if drop_features is None:
            drop_features = ['date', 'title', 'genre', 'thumbnail', 'channel', 'subscribe', 'description']
        train = train_raw.groupby('channel').apply(lambda x: self._to_sequential(x, filter_size, drop_features)).reset_index(drop=True)
        return train
    
    def str_to_datetype(self):
        '''csv파일 로드시 date 컬럼이 str 타입으로 읽혀진 경우 이를 datetype으로 변환'''
        if pd.api.types.is_datetime64_ns_dtype(self.result['date']):
            pass
        else:
            self.result['date'] = pd.to_datetime(self.result['date'])
            
    def add_sub_diff(self):
        '''일간 구독자 변화량 컬럼을 추가하는 함수'''
        self.result = self.result.groupby('channel').apply(lambda x: _get_subs_diff(x)).reset_index(drop=True)
    
    def add_is_upload(self):
        '''해당 날짜에 영상 업로드가 발생했는지(1) 하지않았는지(0)를 담은 변수 생성'''
        self.result = self.result.groupby('channel').apply(lambda x: self._get_is_upload(x)).reset_index(drop=True)
    
    def add_no_upload_interval(self):
        self.result = self.result.groupby('channel').apply(lambda x: self._get_no_upload_interval(x)).reset_index(drop=True)
        
    def add_n_hashtag(self):
        '''영상별 해시태그 개수를 담은 변수 생성(영상 미업로드시 0)'''
        self.result['n_hashtage'] = 0
        self.result.loc[self.result['description'].notnull(), 'n_hashtage'] = \
                (self.result.loc[self.result['description'].notnull(), 'description'].apply(lambda x: len(x.split('#'))-1))
        
    @staticmethod
    def _to_sequential(data, filter_size, drop_features):
        data = data.reset_index(drop=True)
        idx_list = data.index.tolist()
        
        for i in range(len(idx_list)-filter_size):
            sample = data.iloc[idx_list[i:i+filter_size], :].values.reshape(1, -1).flatten()
            sample = np.append(sample, data.loc[idx_list[i+filter_size], 'cumul_subs'])
            train = sample.copy() if i == 0 else np.vstack([train, sample])
        
        train = pd.DataFrame(train, columns = data.columns.tolist() * filter_size + ['target'])
        return train.drop(drop_features, axis=1)
        
    @staticmethod
    def _get_no_upload_interval(data):
        result = data.reset_index(drop=True)
        upload_idx = result[result['is_upload'] == 1].index.tolist()

        temp = [0 for i in range(result.shape[0])]
        for i in range(len(upload_idx)):
            if i == len(upload_idx)-1:
                former = upload_idx[i]
                temp[former+1:] = [i+1 for i in range(len(temp[former+1:]))]
            else:
                former, latter = upload_idx[i], upload_idx[i+1]
                temp[former+1:latter] = [i+1 for i in range(len(temp[former+1:latter]))]
        result['no_upload_interval'] = temp
        return result
    
    @staticmethod
    def _get_is_upload(data):
        result = data.reset_index(drop=True)
        upload_idx = result[result['title'].notnull()].index.tolist()
        result['is_upload'] = 0
        result.loc[upload_idx, 'is_upload'] = 1
        return result
    
    @staticmethod
    def _get_date_diff(data):
        result = data.reset_index(drop=True)
        result['date_diff'] = ((result['date'] - result['date'].shift()).apply(lambda x: x.days)).values
        return result
    
    @staticmethod
    def _get_sub_diff(data):
        result = data.reset_index(drop=True)
        result['sub_diff'] = (result['cumul_subs'] - result['cumul_subs'].shift())
        return result
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:26:29 2020

@author: admin
"""

import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import time
import jieba
from gensim import corpora,models,similarities
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel



class Data_tool:
    
    #连接数据库
    name='etl'
    pw='etl'
    host='192.168.1.191'
    port='3306'
    database='医院销售数据'
    
    # 参数设置
    
    # read_table='dc_unit'#读取表名
    read_table= input('请输入表名，如：dc_unit\n')
    read_column='keyword'#读取字段
    
    # 写入字段
    drug_name='药品名称'
    company_name='企业名称'
    
    # 最佳结果
    res_list='max_list'
    # 可能结果
    res_lists='texts_list'
    
    val=0.7

    def read_texts(self):
        '''
        读取国产进口数据

        Returns
        -------
        texts_list : TYPE
            DESCRIPTION.

        '''
        
        # 用sqlalchemy构建数据库链接engine
        connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(self.name,self.pw,self.host,self.port,self.database)
        engine = create_engine(connect_info)
        
        
        # 读取匹配列表，返回值
        read_texts_list="SELECT DISTINCT `药品名称`,`企业名称`,CONCAT(`药品名称`,`企业名称`) AS `texts_list`FROM be_drugdata;"
        pd_texts=pd.read_sql(sql=read_texts_list,con=engine)
        self.pd_texts=pd_texts
        #转为列表
        texts_list=pd_texts['texts_list'].values.tolist()
     
        print('be_drugdata读取完成{}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        return texts_list






    def read_keywords(self):
        '''
        读取待规范数据表

        Returns
        -------
        keywords : TYPE
            DESCRIPTION.

        '''
        # 用sqlalchemy构建数据库链接engine
        # connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(name,pw,host,port,database)
        connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(self.name,self.pw,self.host,self.port,self.database)
        engine = create_engine(connect_info)
        
        # 对数据进行更新后
        try:
            upsql='''
            UPDATE
            `dc_unit` AS a,
            `文本相似度计算结果` AS b
            SET
            a.`药品名称`=b.`药品名称`,
            a.`企业名称`=b.`企业名称`,
            a.max_list=b.max_list,
            a.texts_list=b.texts_list
            WHERE
            a.keyword=b.keyword;
            '''
            engine.execute(upsql)
            print('dc_unit update 完成！')
        except:
            print('dc_unit update 失败！')
        
        
        
        
        
        # 读取关键字，与列表匹配
        read_keyword="SELECT `keyword` FROM `{}` WHERE `keyword` IS NOT NULL AND `max_list` IS NULL;".format(self.read_table)
        pd_dic=pd.read_sql(sql=read_keyword,con=engine)
        print('{}读取完成{}'.format(self.read_table,time.strftime('%Y-%m-%d %H:%M:%S')))
        # 转为列表
        # keywords = pd_dic['通用名称'].values.tolist()
        keywords = pd_dic['keyword'].values.tolist()
        return keywords



    def to_sparse_matrix(self):
        # 1、将【文本集】生成【分词列表】
        #texts = [lcut(text) for text in texts_list]
        texts = [lcut(str(text)) for text in self.texts_list]
        
        # 2、基于文本集建立【词典】，并获得词典特征数
        dictionary = Dictionary(texts)
        self.dictionary=dictionary
        num_features = len(dictionary.token2id)
        # 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
        corpus = [dictionary.doc2bow(text) for text in texts]
        # 4、创建【TF-IDF模型】，传入【语料库】来训练
        tfidf = TfidfModel(corpus)
        self.tfidf=tfidf
        tf_texts = tfidf[corpus]
        sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
        
        return sparse_matrix




    def fun_tfidf_doc2(self):
    # def fun_tfidf_doc2(keyword,sparse_matrix):
        # global value
    
        # 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
    
        kw_vector = self.dictionary.doc2bow(lcut(self.keyword))
        
    
        # 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
          # 此处将【语料库】用作【被检索文本】
        tf_kw = self.tfidf[kw_vector]
        # 6、相似度计算
        
        similarities = self.sparse_matrix.get_similarities(tf_kw)
    
        #最佳匹配
# =============================================================================
#         similarity=max(similarities)
#         if similarity>self.val:# 判断最大相识度是否满足要求
#             r=np.argmax(similarities, axis=0)
#             # result=pd_texts['me_name'].values[r]
#             result=self.pd_texts[self.texts_list_col].values[r]
#         else:
#             result=None
# =============================================================================
            
        k_list = []
        for e, s in enumerate(similarities):
            if(s >= self.val):
                s_list = []
                s_list.append(self.pd_texts['药品名称'][e])
                s_list.append(self.pd_texts['企业名称'][e])
                s_list.append(self.texts_list[e]) # 关键字
                s_list.append(s) # 分值
                k_list.append(s_list)     
        
        max_list = []
        m=max(similarities)
        for e, s in enumerate(similarities):
            if(s == m):
                s_list = []
                s_list.append(self.pd_texts['药品名称'][e])
                s_list.append(self.pd_texts['企业名称'][e])
                s_list.append(self.texts_list[e]) # 关键字
                s_list.append(s) # 分值
                max_list.append(s_list)            
            
        return k_list,max_list

    def update_tomysql(self):
        # 用sqlalchemy构建数据库链接engine
        # connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(name,pw,host,port,database)
        connect_info = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(self.name,self.pw,self.host,self.port,self.database)
        engine = create_engine(connect_info)
        try:
            # d={'keyword':str(keyword),'药品名称':str(max_list[0][0]),'企业名称':str(max_list[0][1]),'max_list':str(max_list[0]),'texts_list':str(k_list)}
            k_list=self.res[0]
            max_list=self.res[1]
            
            d={'keyword':str(self.keyword),'药品名称':str(max_list[0][0]),'企业名称':str(max_list[0][1]),'max_list':str(max_list[0]),'texts_list':str(k_list)}
            
            dict_df = pd.Series(data=d)
            # df=pd.DataFrame(dict_df)
            df=dict_df.to_frame()
            
            df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            
            
            df2.to_sql('文本相似度计算结果', engine, if_exists='append',index=False)
            print(df2)
        except BaseException as e:
            print(e)
# =============================================================================
#         up_sql='''UPDATE `{}` SET `{}`='{}' WHERE `{}`='{}';'''.format(self.keyword_table,self.keyword_text,self.res,self.keyword_col,self.keyword)
#         try:
#             engine.execute(up_sql)
#         except BaseException as e:
#             print(e)
# =============================================================================
            
    

def run():
    # 读取数据
    datamatch=Data_tool()
    datamatch.texts_list=datamatch.read_texts()
    datamatch.keywords=datamatch.read_keywords()
    
    datamatch.sparse_matrix=datamatch.to_sparse_matrix()
    
    datamatch.progress=0
    for keyword in datamatch.keywords:
        datamatch.progress +=1
        datamatch.keyword=keyword
        datamatch.res=None
        datamatch.res=datamatch.fun_tfidf_doc2()
        if datamatch.res:
            datamatch.update_tomysql()
        else:
            print('{}无相似度大于{}的结果!{}'.format(keyword,datamatch.val,time.strftime('%Y-%m-%d %H:%M:%S')))
        print('进度{:.2f}%'.format(datamatch.progress/len(datamatch.keywords)*100))

run()    
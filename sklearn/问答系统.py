import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from helper.text_normalizer import normalize_corpus
from helper.text_normalizer import normalize_text

class QA_Model:
    def __init__(self):
        self.qlist,self.alist = self.read_corpus()
        self.norm_qlist = normalize_corpus(self.qlist)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_qlist,self.searchTable = self.buildSearchTable()


    def read_corpus(self):
        with open("../../MyDataSets/问答数据/small_train-v2.0.json", "r") as f:
            load_dict = json.load(f)
        qlist = []
        alist = []
        for i in load_dict['data']:
            for j in i['paragraphs']:
                for k in j['qas']:
                    qlist.append(k['question'])
                    for p in k['answers']:
                        alist.append(p['text'])
        qlist = np.array(qlist)
        alist = np.array(alist)
        return qlist, alist

    def cos_dist(self,vec1, vec2):
        """
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 返回两个向量的余弦相似度
        """
        vec2 = vec2.T
        similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return similarity

    def buildSearchTable(self):
        tfidf_qlist = self.vectorizer.fit_transform(self.norm_qlist)
        tfidf_qlist = tfidf_qlist.toarray()
        dic = {}
        for word in self.vectorizer.get_feature_names():
            dic[word] = []
            for i in range(len(self.norm_qlist)):
                if word in self.norm_qlist[i]: dic[word].append(i)
        return tfidf_qlist,dic

    def getSearhIndex(self,input_q):
        norm_ask = normalize_text(input_q)
        searchIndex = set()
        for word in norm_ask:
            if word in self.searchTable:
                searchIndex = searchIndex | set(self.searchTable[word])
        return list(searchIndex)

    def top5result(self,input_q,n=5):
        if n <1 : n = 1
        searchIndex = self.getSearhIndex(input_q)
        print("共需要匹配%d个问题"%len(searchIndex))
        asklist = []
        asklist.append(input_q)
        norm_ask = normalize_corpus(asklist)
        tfidf_ask = self.vectorizer.transform(norm_ask)
        esask = tfidf_ask[0].toarray()
        all_hits = []  # 余弦相似度
        for sentence in self.tfidf_qlist[searchIndex]:
            all_hits.append(self.cos_dist(esask, sentence))
        all_hits = np.array(all_hits)
        all_index = np.array(np.argsort(all_hits))

        searchIndex = np.array(searchIndex)
        top5_index = searchIndex[all_index[:-(n+1):-1]]
        print("top%d相似度:\n%s"%(n,all_hits[all_index[:-(n+1):-1]]))

        print("匹配到top%d的问题:\n%s"%(n,self.qlist[top5_index]))
        print("对应到top%d的答案:\n%s"%(n,self.alist[top5_index]))
        return self.qlist[top5_index],self.alist[top5_index]


def main():
    '''
        1.读取语料库
        2.数据预处理
        3.搭建模型
            1)文本表示：TF-IDF
        4.开始使用
            1)输入文本的预处理
            2)文本表示：TF-IDF
            3)计算与语料库中每一个问题的余弦相似度
            4)找出相似度最高的top5问题的答案
    '''
    input_q="Where did Destiny's Child end their group act?"
    print("用户输入的问题:\n",input_q)
    model = QA_Model()
    model.top5result(input_q,n=5)
    return  None;

if __name__ == "__main__":
    main()
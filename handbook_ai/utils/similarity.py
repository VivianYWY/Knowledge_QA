from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDF_similarity(text1, text2):
    # 使用TF-IDF向量化器将文本转换为向量
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return cosine_sim[0][0]

def jaccard_similarity(text1, text2):
    # 将文本转换为词汇集合
    list1 = set(text1.split())
    list2 = set(text2.split())

    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection

    return intersection / float(union)


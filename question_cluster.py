import json
from pymongo import MongoClient
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def get_mongo_client():
    password = 'hf6Wbg3dm8'
    url = 'mongodb://root:'+password+'@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719'
    client = MongoClient(url)
    print("Connected to MongoDB")
    return client

# 编码函数
def encode_text(tokenizer,bert_model,text):
    # BERT模型和tokenizer初始化
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # 使用[CLS] token的输出作为文本的表示
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 计算距离 (x1 + x2 + x3) / 3
def calculate_distance_matrix(context_embeddings, learning_objective_embeddings, length_embeddings):
    # 计算每个特征的余弦相似度
    context_sim = cosine_similarity(context_embeddings)
    learning_obj_sim = cosine_similarity(learning_objective_embeddings)
    length_sim = cosine_similarity(length_embeddings)

    # 计算距离
    context_distance = 1 - context_sim
    learning_obj_distance = 1 - learning_obj_sim
    length_distance = 1 - length_sim

    # 综合距离 (x1 + x2 + x3) / 3
    total_distance = (context_distance + learning_obj_distance + length_distance) / 3
    return total_distance

if __name__ == '__main__':
    client = get_mongo_client()
    db = client['pro_mathaday_assessment']
    collection = db["assessment_questions"]
    
    questions = []
    # 遍历table中所有的文档，提取itemId,description['en'],mathObjectives
    for doc in collection.find():
        itemId = doc['itemId']
        description = doc['description']['en']
        mathObjectives = doc['mathObjectives']
        mathObjectives = ','.join(mathObjectives)
        questions.append({'itemid': itemId, 'question_context': description, 'learning_objective': mathObjectives, 'question_length': len(description.split())})
    
    print(len(questions))
    # print(questions)   

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    print("BERT model loaded")
    
    # 对所有问题进行编码
    context_embeddings = []
    learning_objective_embeddings = []
    length_embeddings = []

    for question in questions:
        # 编码每个特征
        context_emb = encode_text(tokenizer,bert_model,question['question_context'])
        learning_obj_emb = encode_text(tokenizer,bert_model,question['learning_objective'])
        length_emb = np.array([question['question_length']])  # 作为一个标量嵌入

        context_embeddings.append(context_emb)
        learning_objective_embeddings.append(learning_obj_emb)
        length_embeddings.append(length_emb)

    # print(context_embeddings)
    
    # 转换为numpy数组
    context_embeddings = np.array(context_embeddings)
    learning_objective_embeddings = np.array(learning_objective_embeddings)
    length_embeddings = np.array(length_embeddings)

    # 计算所有问题的综合距离
    distance_matrix = calculate_distance_matrix(context_embeddings, learning_objective_embeddings, length_embeddings)

    # print(distance_matrix)
    
    # 层次聚类
    clustering = AgglomerativeClustering(linkage='complete', metric='precomputed', n_clusters=30)
    cluster_labels = clustering.fit_predict(distance_matrix)

    
    # 按照itemid输出聚类结果
    itemid_to_cluster = {}
    for idx, question in enumerate(questions):
        itemid_to_cluster[question['itemid']] = cluster_labels[idx]

    # 输出一共有多少个聚类
    print(f"Number of clusters: {len(set(cluster_labels))}")
    
    # 找到同一聚类中的问题
    clustered_questions = {}
    for itemid, cluster in itemid_to_cluster.items():
        if cluster not in clustered_questions:
            clustered_questions[cluster] = []
        clustered_questions[cluster].append(itemid)

    # 为每个聚类创建一个json文件，文件名为cluster_{cluster_id}.json, 内容为该聚类中的itemid
    # 存储在 data/clustered_questions 目录下
    for cluster, itemids in clustered_questions.items():
        with open(f'data/clustered_questions/cluster_{cluster}.json', mode='w') as file:
            json.dump(itemids, file)

    
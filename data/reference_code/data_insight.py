from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from mysql import *
from datetime import datetime
import csv

def get_mongo_client():
    password = 'hf6Wbg3dm8'
    url = 'mongodb://root:'+password+'@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719'
    client = MongoClient(url)
    print("Connected to MongoDB")
    return client

if __name__ == '__main__':
    client = get_mongo_client()
    db = client['stg_mathaday_assessment']
    collection = db['assessment_objective_performances']
    # # 统计所有的userid
    # user_info = {user: [] for user in collection.distinct('userId')}
    # print(len(user_info))
    # # 统计每个人对应的document数量
    # for id in user_info:
    #     user_info[id] = collection.count_documents({'userId': id})
    # # 输出数量前5的userid和document数量:  [(48995, 23057), (49152, 12572), (49162, 8432), (48983, 6063), (49011, 4284)]
    # print(sorted(user_info.items(), key=lambda x: x[1], reverse=True)[:5])
   
    # userId = 49011
    # 统计userId=49011每个objective的正确率，平均做题时间
    result = collection.find({'userId': 49011})
    objective_info = {obj: [] for obj in result.distinct('mathObjective')}
    print("len of objective_info:",len(objective_info))

    # 找到每个objective下number of questions最大的项
    for obj in objective_info:
        result = collection.find({'userId':49011,'mathObjective': obj})
        number_of_questions = result.distinct('numberOfQuestions')
        objective_info[obj] = {"number_of_questions":max(number_of_questions)}
    # print(objective_info)

    # 统计每个objective的createdAt
    for obj in objective_info:
        result = collection.find({'mathObjective': obj})
        created_at = result.distinct('createdAt')
        objective_info[obj]['created_at'] = created_at

    result = collection.find({'userId': 49011, 'mathObjective': 'M1001'})
    # result = collection.find({'userId': 48983, 'mathObjective': 'M1007'})
    created_at = result.distinct('createdAt')
    # 转换成日期
    dates = [datetime.fromtimestamp(ts) for ts in created_at]
    # 转换成年月日小时秒
    dates = [date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]
    # dates = [date.strftime('%Y-%m-%d') for date in dates]
    # print(dates)


    # 存储每个objective的做题数量，分数，做题花费时间，创建时间
    with open('objective_info.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['objective', 'number_of_questions', 'average_score', 'average_time_spent', 'created_at'])
        for obj in objective_info:
            result = collection.find({'userId':49011,'mathObjective': obj})
            # count = collection.count_documents({'userId':49011,'mathObjective': obj})
            # 创建一个数组存储每个objective的createdAt
    
            dates = {datetime.fromtimestamp(date).strftime('%Y-%m-%d'): [] for date in result.distinct('createdAt')}
            for doc in result:
                number_of_questions = doc['numberOfQuestions']
                totalScore = doc['totalScore']
                totalTimeSpent = doc['totalTimeSpent']
                created_at = datetime.fromtimestamp(doc['createdAt']).strftime('%Y-%m-%d') 
                # 计算日期相同的平均分数和平均时间
                if created_at in dates:
                    dates[created_at].append(totalScore)
                    dates[created_at].append(totalTimeSpent)
                    dates[created_at].append(number_of_questions) 
                else:
                    dates[created_at] = [totalScore, totalTimeSpent, number_of_questions]

            # 计算每个objective的平均分数和平均时间
            for date in dates:
                number_of_questions = sum(dates[date][2::3])
                average_score = sum(dates[date][0::3]) / number_of_questions
                average_time_spent = sum(dates[date][1::3]) / number_of_questions
                writer.writerow([obj, number_of_questions, average_score, average_time_spent, date])
    
        
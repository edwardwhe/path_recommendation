from pymongo import MongoClient
import json

def get_mongo_client():
    password = 'hf6Wbg3dm8'
    url = 'mongodb://root:'+password+'@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719'
    client = MongoClient(url)
    print("Connected to MongoDB")
    return client

if __name__ == '__main__':
    client = get_mongo_client()
    db = client['pro_mathaday_assessment']
    collection = db['assessments']
    # 找到syllabi[0]['unitId'], 存入的是assessmentId
    dict = {}
    for doc in collection.find():
        unit_id = doc['syllabi'][0]['unitId']
        if unit_id not in dict:
            dict[unit_id] = []
        else:
            dict[unit_id].append(doc['id'])
    # print(dict)
            
    for key in dict:
        with open('data/assessment_questions/' + str(key) + '.json', mode='w') as file:
            json.dump(dict[key], file)
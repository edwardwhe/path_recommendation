import numpy as np
from data_insight import get_mongo_client

if __name__ == '__main__':
    client = get_mongo_client()
    db = client['stg_mathaday_assessment']
    collection = db['assessment_question_records']
    result = collection.find({'userId': 49011})
    assessmentQuestionId = [doc['assessmentQuestionId'] for doc in result]
    objectivelist = ["N1001","N1002","N1003","N1004","N1005","N1006","N1007","N1008","N1009"]
    print(len(assessmentQuestionId))
    
    # 得到题目id对应的难度
    difficulty = {}
    for id in assessmentQuestionId:
        result = db["assessment_questions"].find({"itemId": id})
        for res in result:
            difficulty_value = res['difficulty']   
            if difficulty_value is None:
                difficulty_value = "NULL" 
            difficulty[id] = difficulty_value
    
    print(len(difficulty))      
    
    # 得到题目id对应的objectves(0-9)
    # [因为目前只选取了10个objectieve,所有很多id对应的objectives为空]
    objective = {}
    for id in assessmentQuestionId:
        objective[id] = []
        result = db["assessment_questions"].find({"itemId": id})
        for res in result:
            mathObjectives = res['mathObjectives'] 
            for obj in mathObjectives:
                if obj in objectivelist:
                    index = objectivelist.index(obj)
                    objective[id].append(index)
    print(len(objective))
                      
    # 得到每个objective对应的题目id，分数，和做题时间
    info = {}
    for id in assessmentQuestionId:
        result = db["assessment_question_records"].find({"assessmentQuestionId": id,"userId": 49011})
        for doc in result:
            score = doc['score']
            updatedAt = doc['updatedAt']
            info[id] = [{"score": score, "updatedAt": updatedAt}]
    print(len(info))                  
    
   
    # 计算每个objective的平均分
    obj_score = {}
    for id in info:
        obj = objective[id]
        if len(obj) == 0:
            continue
        else:
            for i in obj:
                if i not in obj_score:
                    obj_score[i] = []
                obj_score[i].append(info[id][0]['score'])
    
    
    avg_score = {}
    for obj in obj_score:
        avg_score[obj] = sum(obj_score[obj])/len(obj_score[obj])
    
    # 选定10个objective进行计算 N1001-N1009[database:objective_performance]
    profile = [0]*10
    for obj in avg_score:
        id = int(obj)
        profile[id] = avg_score[obj]
                
    
    
    # 将info中的数据按照时间顺序排列
    for item in info:
        info[item].sort(key=lambda x: x['updatedAt'])
    
    # info中加入objective信息
    for id in info:
        info[id][0]['objective'] = objective[id]
        
               
    features = []
    target = []
    for item in info:
        vector = [0] * 10
        obj = info[item][0]['objective']
        length = len(obj)
        if length != 0:
            for i in obj:
                vector[i] = 1
                
        difficulty_value = difficulty[item]
        if difficulty_value == "easy":
            vector.extend([1,0,0])
        elif difficulty_value == "medium":
            vector.extend([0,1,0])
        elif difficulty_value == "hard":
            vector.extend([0,0,1])
        else:
            vector.extend([0,0,0])
        features.append(vector)
        
        if info[item][0]['score'] == 100:
            target.append(1)
        else:
            target.append(0)
        
  
    
    print(np.array(profile).shape)
    print(np.array(features).shape)
    print(np.array(target).shape)
    
    np.save("profile.npy",profile)  
    np.save("features.npy",features)
    np.save("labels.npy",target)
    

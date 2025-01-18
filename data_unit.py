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
    collection = db['assessment_questions']
    # 读取data/obj_to_unit.json文件
    with open('data/obj_to_unit.json', mode='r') as file:
        obj_to_unit = json.load(file)
    
    unit_dict = {}
    # 收集所有不重复的unit值并初始化字典
    unique_units = set()
    for units in obj_to_unit.values():
        if isinstance(units, (list, tuple)):
            unique_units.update(units)
        else:
            unique_units.add(units)
            
    for unit in unique_units:
        unit_dict[unit] = []
        
    for item in collection.find():
        itemId = item['itemId']
        mathObjectives = item['mathObjectives']
        for obj in mathObjectives:
            if obj in obj_to_unit:
                unit = obj_to_unit[obj]
                # 如果unit是列表或元组
                if isinstance(unit, (list, tuple)):
                    for u in unit:
                        if itemId not in unit_dict[u]:
                            unit_dict[u].append(itemId)
                else:
                    if itemId not in unit_dict[unit]:
                        unit_dict[unit].append(itemId)
                    
    # 如果unit_dict中的值为空列表，则删除该键
    for key in list(unit_dict.keys()):
        if not unit_dict[key]:
            del unit_dict[key]
            
    # 将unit_dict写入文件，每个unit存一个文件
    for key in unit_dict:
        with open('data/assessment_questions/' + str(key) + '.json', mode='w') as file:
            json.dump(unit_dict[key], file)
    
                
                
    
    
    
            

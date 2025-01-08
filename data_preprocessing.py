from pymongo import MongoClient
import csv
from type import *
import numpy as np
import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt
from path_recommendation import group_students

class DataPreprocessing:
  
  def __init__(self):
    self.client = MongoClient("mongodb://root:hf6Wbg3dm8@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719")
    self.db = self.client['pro_mathaday_assessment']
    self.question_records = self.db['assessment_question_records']

    self.userIds = self.load_user_ids()
    self.questionIds = self.load_question_ids()

  def load_user_ids(self):
    user_ids = self.question_records.distinct('userId')
    self.save_json(user_ids, "data/user_id.json")
    return user_ids

  def load_question_ids(self):
    question_ids = self.question_records.distinct('assessmentQuestionId')
    self.save_json(question_ids, "data/assessment_question_id.json")
    return question_ids

  def load_assessments(self):
    assessments_table = dataPreprocessing.db["assessments"]
    assessments = assessments_table.find({})
    assessment_info = {}
    for assessment in tqdm.tqdm(assessments):
      assessment_info[assessment["id"]] = {
        "type": assessment["type"],
        "syllabi": [syllable.get("unitId") for syllable in assessment.get("syllabi")]
      }

    self.save_json(assessment_info, "data/assessment_info.json")
    return assessment_info

  def load_assessment_questions(self):
    assessment_questions_table = dataPreprocessing.db["assessment_questions"]
    assessment_questions = assessment_questions_table.find({})
    assessment_question_info = {}
    for assessment_question in tqdm.tqdm(assessment_questions):
      assessment_question_info[assessment_question["itemId"]] = {
        "difficulty": self.convert_difficulty_to_num(assessment_question["difficulty"]),
        "grade": assessment_question["grade"]
      }
    self.save_json(assessment_question_info, "data/assessment_question_info.json")
    return assessment_question_info

  def convert_difficulty_to_num(self, difficulty):
    if difficulty == "easy":
      return 1
    elif difficulty == "medium":
      return 2
    elif difficulty == "hard":
      return 3
    else:
      # If the difficulty is not easy, medium or hard, return 1
      return 1

  def load_json(self, file_name):
    # Load a JSON file and return the data
    with open(file_name, mode='r') as file:
      data = json.load(file)
    return data
  
  def save_json(self, data, file_name):
    # Save data to a JSON file
    with open(file_name, mode='w') as file:
      json.dump(data, file)

  # deprecated
  def load_csv(self, file_name):
    # Read the CSV file and load it back as a list of dictionaries
    with open(file_name, mode='r', encoding='utf-8') as file:
      reader = csv.reader(file)
      rows = [row for row in reader]  # List of dictionaries
    return rows

  def load_user_data(self, user_id):
    user_data = self.question_records.find({"userId": user_id})
    if not user_data:
      raise ValueError(f"No data found for userId {user_id}")
    
    assessments_info = self.load_json("data/assessment_info.json")
    # Parse assessments and questions
    assessments = []
    

    for record in user_data:
      # Get the assessment to get corresponding unit ID, type and assessmentID.
      assessment = assessments_info.get(record["assessmentId"])
      # if cannot find the assessment, skip this record
      if not assessment:
        continue
      # Get the unit ID of the syllabus
      assessment_question = Assessment(
        start_time=record["createdAt"],
        duration=record.get("totalTimeSpent"),
        score=record.get("score"),
        question_id=record.get("assessmentQuestionId"),
        assessment_id=record.get("assessmentId"),
        assessment_type=assessment.get("type"),
        unit=assessment.get("syllabi")
      )
      assessments.append(assessment_question)

    # Create User object
    user = User(
      user_id=user_id,
      assessments=assessments
    )
    return user

  def get_user_data_from_database(self):
    users = []  # declare users to be list of User type
    for user_id in tqdm.tqdm(self.userIds):
      user = self.load_user_data(user_id)
      users.append(user)
    return users
    
    

    # # save the result to a csv file
    # csv_file_path = "total_time_spent.csv"
    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #   writer = csv.writer(file)
    #   writer.writerows(map(lambda x: [x], total_time_spent))

  def get_data_from_database(self):

    # # save the result to a file that can be loaded in the future
    # # the user is not JSON serializable, so we need to convert it to a dictionary
    users = self.get_user_data_from_database()
    users_dict = [user.__dict__ for user in users]
    self.save_json(users_dict, "data/users.json")  

  def label_data(self):
    json_file_path = "data/users.json"
    with open(json_file_path, mode='r') as file:
      user_data = json.load(file)
    # each sublist is a user's average score, intensity, regularity, and total time spent
    user_data = self.clean_data(user_data)
    user_cluster_vector = []
    for key, data in user_data.items():
      total_time_spent = int(data['time_spent']/10000)
      user_cluster_vector.append([int(data['score']), int(data['intensity']), int(data['regularity']), total_time_spent])
    means, stds, labels = group_students(user_cluster_vector, 3)
    
    # Add label field to user_data
    for i, (key, data) in enumerate(user_data.items()):
      data['label'] = float(labels[i])
    
    # Save clustered user data to a new JSON file
    clustered_json_file_path = "clustered_user_data.json"
    with open(clustered_json_file_path, mode='w') as file:
      json.dump(user_data, file)
    print(user_data)
    return user_cluster_vector
  
  def get_data(self):
    json_file_path = "clustered_user_data.json"
    with open(json_file_path, mode='r') as file:
      user_data = json.load(file)
    cluster_user_0 = []
    for key, data in user_data.items():
      if data['label'] == 1:
        cluster_user_0.append(self.get_matrix(data['assessment_id']))
    return cluster_user_0

  def clean_data(self, user_data):
    # Clean the data
    user_data = {key: data for key, data in user_data.items() if data['score'] != 0 and data['score'] != 100 and data['time_spent'] / 10000 != 0}
    return user_data
  
  def get_matrix(self, assessment_ids):
    # Convert the sorted order into a matrix using numpy
    assessment_matrix = np.zeros((len(assessment_ids), len(self.questionIds)))
    tracking_vector = np.zeros(len(self.questionIds))
    for i, id in enumerate(assessment_ids):
      question_index = self.questionIds.index([id])
      tracking_vector[question_index] = 1 if tracking_vector[question_index] == 0 else 0
      assessment_matrix[i] = tracking_vector
    return assessment_matrix
    
if __name__ == "__main__":
  dataPreprocessing = DataPreprocessing()
  dataPreprocessing.performance_analysis()
  # assessments_table = dataPreprocessing.db["assessments"]
  # result = assessments_table.find_one({'id': "21c6b9e9-8f78-4700-b08e-03ebf645caf2"})
  # syllabi = result.get("syllabi")
  # units = [syllabus.get("unitId") for syllabus in syllabi]
  # print(units)

  
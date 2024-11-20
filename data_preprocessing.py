from pymongo import MongoClient
import csv
from type import *
import numpy as np
import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt

class DataPreprocessing:
  
  def __init__(self):
    self.client = MongoClient("mongodb://root:hf6Wbg3dm8@dds-3ns35de5eee23e941756-pub.mongodb.rds.aliyuncs.com:3717,dds-3ns35de5eee23e942366-pub.mongodb.rds.aliyuncs.com:3717/admin?replicaSet=mgset-33008719")
    self.db = self.client['stg_mathaday_assessment']
    self.question_records = self.db['assessment_question_records']
    self.assessments = self.db['assessments']

    self.userIds = self.load_csv("user_id.csv")
    self.questionIds = self.load_csv("assessment_question_id.csv")

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
    
    # Parse assessments and questions
    assessments = []
    for record in user_data:
      assessment = Assessment(
        start_time=record["createdAt"],
        duration=record.get("totalTimeSpent"),
        score=record.get("score"),
        question=record.get("assessmentQuestionId")
      )
      assessments.append(assessment)

    # Create User object
    user = User(
      user_id=user_id,
      assessments=assessments,
      questionIds=self.questionIds
    )
    return user

  def get_user_data_from_database(self):
    users = []  # declare users to be list of User type
    for user_id in tqdm.tqdm(self.userIds):
      user = self.load_user_data(int(user_id[0]))
      users.append(user)
    return users
    
    # average_scores = [sum(user.get_score_vector()) / len(user.get_score_vector()) for user in users]
    # total_time_spent = [sum(user.get_duration_vector()) for user in users]
    # start_times = [user.get_start_time() for user in users]

    # # save the result to a csv file
    # csv_file_path = "total_time_spent.csv"
    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #   writer = csv.writer(file)
    #   writer.writerows(map(lambda x: [x], total_time_spent))

  def get_data_from_database(self):
    users = self.get_user_data_from_database()
    total_score = self.load_csv("total_score.csv")
    total_time_spent = self.load_csv("total_time_spent.csv")
    start_times = self.load_csv("start_times.csv")
    # start_times = [datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S") for time in start_times]
    start_times = [eval(time[0]) for time in start_times]
    assessment_ids = [user.get_assessment_ids() for user in users]

    # convert the start time to a list of datetime objects
    start_time_list = []
    for i, time in enumerate(start_times):
      start_time_list.append([datetime.fromtimestamp(t) for t in time])

    # count the number of submission of each week, the result is saved as [[week number 1, week number 2, ...], [week number 1, week number 2, ...], ...]
    week_numbers_ = [[time.isocalendar()[1] for time in times] for times in start_time_list]

    # count the number of submission of each week, the result is saved as [(user 1)[(week number 1, number of submissions in this week), (week number 1, number of submissions in this week)...], (user 2)[...]]
    submission_counts = []
    for i in range(len(start_times)):
      user_submissions = []
      for week_number in set(week_numbers_[i]):
        count = week_numbers_[i].count(week_number)
        user_submissions.append((week_number, count))
      submission_counts.append(user_submissions)

    # get the average number of submission of each user, the time span is 1 week
    user_intensitys = []
    for i in range(len(submission_counts)):
      user = submission_counts[i]
      counts = 0
      for submission in user:
        counts += submission[1]
      user_intensitys.append(counts/len(submission_counts[i]))
    print(user_intensitys)

    # get the regularity of each user, which is the number of weeks that the user has submission
    user_regularitys = []
    for i in range(len(submission_counts)):
      user_regularitys.append(len(submission_counts[i]))
    print(user_regularitys)

    user_data = {}
    for i in range(len(self.userIds)):
      user_id = int(self.userIds[i][0])
      start_time = start_times[i]
      intensity = user_intensitys[i]
      regularity = user_regularitys[i]
      score = float(total_score[i][0])
      time_spent = float(total_time_spent[i][0])
      assessment_id = assessment_ids[i]
      
      user_data[user_id] = {
        'assessment_id': assessment_id,
        'start_time': start_time,
        'intensity': intensity,
        'regularity': regularity,
        'score': score,
        'time_spent': time_spent
      }
    # Save data_dict to a JSON file
    json_file_path = "user_data.json"
    with open(json_file_path, mode='w') as file:
      json.dump(user_data, file)

    # Reload data_dict from the JSON file
    with open(json_file_path, mode='r') as file:
      reloaded_data_dict = json.load(file)

    print(reloaded_data_dict)
    print(user_data)
  
  # def label_data(self):
  #   json_file_path = "user_data.json"
  #   with open(json_file_path, mode='r') as file:
  #     user_data = json.load(file)
  #   # each sublist is a user's average score, intensity, regularity, and total time spent
  #   user_data = self.clean_data(user_data)
  #   user_cluster_vector = []
  #   for key, data in user_data.items():
  #     total_time_spent = int(data['time_spent']/10000)
  #     user_cluster_vector.append([int(data['score']), int(data['intensity']), int(data['regularity']), total_time_spent])
  #   means, stds, labels = group_students(user_cluster_vector, 3)
    
  #   # Add label field to user_data
  #   for i, (key, data) in enumerate(user_data.items()):
  #     data['label'] = float(labels[i])
    
  #   # Save clustered user data to a new JSON file
  #   clustered_json_file_path = "clustered_user_data.json"
  #   with open(clustered_json_file_path, mode='w') as file:
  #     json.dump(user_data, file)
  #   print(user_data)
  #   return user_cluster_vector
  
  def get_data(self):
    json_file_path = "clustered_user_data.json"
    with open(json_file_path, mode='r') as file:
      user_data = json.load(file)
    cluster_user_0 = []
    for key, data in user_data.items():
      if data['label'] == 0:
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
  dataPreprocessing.get_data()
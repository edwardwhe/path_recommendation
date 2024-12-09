from ivan_util import util
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

class DataAnalysis:
    def performance_analysis(self):
      # load user data
      users = util.load_json(util, "data/users.json")
      assessment_question_info = util.load_json(util, "data/assessment_question_info.json")
      user_scores = []
      for user in users:
        if not user["assessments"]:
          continue
        total_num = len(user["assessments"])
        weighted_total_score = 0
        for assessment in user["assessments"]:
          # if assessment.get("assessment_type") == 1 or assessment.get("assessment_type") == 8 or assessment.get("assessment_type") == 9: 
          if assessment.get("assessment_type") == 5 or assessment.get("assessment_type") == 7: 
            continue
          difficulty = assessment_question_info[assessment["question"]]["difficulty"]
          weighted_total_score += assessment["score"] * (1 + (difficulty - 1) * 0.2)
        average_score = weighted_total_score / total_num
        user["weighted_average_score"] = average_score
        user_scores.append({"user_id": user['user_id'], "score": average_score})
      print(np.mean([user["score"] for user in user_scores]))
      util.save_json(util, user_scores, "data/user_scores.json")
    
    # A: 90, B: 80, C: 70, D: 60, F: the score is less than 60
    # write a function to classify the students into three categories: A, B, C, D, F
    # based on the weighted average score 
    def classify_student(self):
      user_scores = util.load_json(util, "data/user_scores.json")
      user_cluster = []
      for key, user in enumerate(user_scores):
        if not user.get('score'):
          continue
        score = user.get('score')
        if score >= 90:
          user['classification'] = "A"
        elif score >= 80:
          user['classification'] = "B"
        elif score >= 70:
          user['classification'] = "C"
        elif score >= 60:
          user['classification'] = "D"
        else:
          user['classification'] = "F"
        user_cluster.append({"user_id": user['user_id'], "classification": user['classification']})
      util.save_json(util, user_cluster, "data/user_cluster.json")
    
    def plot_student_classification(self):
      user_cluster = util.load_json(util, "data/user_cluster.json")
      user_cluster.sort(key=lambda x: x["classification"])
      classification = [user["classification"] for user in user_cluster]
      plt.hist(classification, bins=5)
      plt.xlabel("Classification")
      plt.ylabel("Number of Students")
      plt.title("Student Classification")
      plt.show()

    def behavior_analysis(self):
      # load user data
      users = util.load_json(util, "data/users.json")
      total_time_spent = []
      for user in users:
        if not user["assessments"]:
          continue
        current_total_time_spent = 0
        for assessment in user["assessments"]:
          current_total_time_spent += assessment["duration"]
        total_time_spent.append({"user_id": user['user_id'], "total_time_spent": current_total_time_spent})
      util.save_json(util, total_time_spent, "data/total_time_spent.json")
      
      start_times = []
      for user in users:
        if not user["assessments"]:
          continue
        current_start_times = []
        for assessment in user["assessments"]:
          current_start_times.append(assessment["start_time"])
        start_times.append({"user_id": user['user_id'], "start_times": current_start_times})
      util.save_json(util, start_times, "data/start_times.json")

      # start_times = [datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S") for time in start_times]
      # assessment_ids = [user.get_assessment_ids() for user in users]

      # # convert the start time to a list of datetime objects
      # start_time_list = []
      # for i, time in enumerate(start_times):
      #   start_time_list.append([datetime.fromtimestamp(t) for t in time])

      # # count the number of submission of each week, the result is saved as [[week number 1, week number 2, ...], [week number 1, week number 2, ...], ...]
      # week_numbers_ = [[time.isocalendar()[1] for time in times] for times in start_time_list]

      # # count the number of submission of each week, the result is saved as [(user 1)[(week number 1, number of submissions in this week), (week number 1, number of submissions in this week)...], (user 2)[...]]
      # submission_counts = []
      # for i in range(len(start_times)):
      #   user_submissions = []
      #   for week_number in set(week_numbers_[i]):
      #     count = week_numbers_[i].count(week_number)
      #     user_submissions.append((week_number, count))
      #   submission_counts.append(user_submissions)

      # # get the average number of submission of each user, the time span is 1 week
      # user_intensitys = []
      # for i in range(len(submission_counts)):
      #   user = submission_counts[i]
      #   counts = 0
      #   for submission in user:
      #     counts += submission[1]
      #   user_intensitys.append(counts/len(submission_counts[i]))
      # print(user_intensitys)

      # # get the regularity of each user, which is the number of weeks that the user has submission
      # user_regularitys = []
      # for i in range(len(submission_counts)):
      #   user_regularitys.append(len(submission_counts[i]))
      # print(user_regularitys)

      # user_data = {}
      # for i in range(len(self.userIds)):
      #   user_id = self.userIds[i]
      #   start_time = start_times[i]
      #   # intensity = user_intensitys[i]
      #   # regularity = user_regularitys[i]
      #   score = float(average_scores[i])
      #   time_spent = float(total_time_spent[i])
      #   assessment_id = assessment_ids[i]
        
      #   user_data[user_id] = {
      #     'assessment_id': assessment_id,
      #     'start_time': start_time,
      #     # 'intensity': intensity,
      #     # 'regularity': regularity,
      #     'score': score,
      #     'time_spent': time_spent
      #   }

      # self.save_json(user_data, "data/user_data.json")
      # reloaded_data_dict = self.load_json("data/user_data.json")

      # print(reloaded_data_dict)
      # print(user_data)


if __name__ == "__main__":
    data_analysis = DataAnalysis()
    data_analysis.behavior_analysis()
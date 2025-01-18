from ivan_util import util
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from matplotlib.sankey import Sankey
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# 包括所有的Analysis Tasks
# 学生的成绩分布情况
# 学生的行为分析
# Markov可以接受的数据格式的转换
class DataAnalysis:

    # Deprecated
    # 学生的成绩的分析
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
          # 1, 1.2 1.4
          weighted_total_score += assessment["score"] * (1 + (difficulty - 1) * 0.2)
        average_score = weighted_total_score / total_num
        user["weighted_average_score"] = average_score
        user_scores.append({"user_id": user['user_id'], "score": average_score})
      print(np.mean([user["score"] for user in user_scores]))
      util.save_json(util, user_scores, "data/user_scores.json")
    
    # Deprecated
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
    
    ############################################################  
    # 学生的行为分析
    # 行为分析的指标：
    # 把总时长拆分成两个指标：强度和规律性
    # Intensity强度：每周平均做题的次数
    # Regularity规律性：一共做题的周数
    def behavior_analysis(self):
      # load user data
      users = util.load_json(util, "data/users.json")
  
      # 计算每个用户的强度和规律性
      # calculate total time spent, intensity and regularity of each user
      user_daily_performance = []
      for user in users:
        if not user["assessments"]:
          continue
        current_start_times = []
        current_total_time_spent = 0
        for assessment in user["assessments"]:
          # 排除考试题目
          if assessment.get("assessment_type") == 5 or assessment.get("assessment_type") == 7: 
            continue
          # 所有做题时间小于2s或者大于93s的数据都是异常数据，需要排除
          if assessment.get("duration") < 2000 or assessment.get("duration") > 93500:
            continue
          current_start_times.append(assessment["start_time"])
          current_total_time_spent += assessment["duration"]
        if not current_start_times:
          continue
        # 提取每周的做题次数
        current_start_times = [datetime.fromtimestamp(time) for time in current_start_times]
        week_numbers = [time.isocalendar()[1] for time in current_start_times]
        week_counts = {}
        for week in week_numbers:
          if week not in week_counts:
            week_counts[week] = 0
          week_counts[week] += 1
        intensity = sum(week_counts.values()) / len(week_counts)
        regularity = len(week_counts)
        user_daily_performance.append({"user_id": user['user_id'], "intensity": intensity, "regularity": regularity, "total_time_spent": current_total_time_spent})
      # Normalize each column data of each user other than user_id
      if user_daily_performance:
          df = pd.DataFrame(user_daily_performance)
          normalized_df = df.copy()
          for column in ['intensity', 'regularity', 'total_time_spent']:
            median = df[column].median()
            mad = df[column].mad()
            normalized_df[column] = (df[column] - median) / mad
          user_daily_performance = normalized_df.to_dict(orient='records')
          
      # Test强度和规律性的相关性
      intensity = df['intensity']
      regularity = df['regularity']
      correlation, p_value = pearsonr(intensity, regularity)
      print(f"Pearson correlation coefficient: {correlation}")
      print(f"P-value: {p_value}")
      if p_value < 0.05:
          print("The correlation between intensity and regularity is statistically significant.")
      else:
          print("The correlation between intensity and regularity is not statistically significant.")

      # 基础的关于强度和规律性的统计信息
      intensity_quantiles = df['intensity'].quantile([0.05, 0.5, 0.95])
      regularity_quantiles = df['regularity'].quantile([0.05, 0.5, 0.95])
      total_time_spent_quantiles = df['total_time_spent'].quantile([0.05, 0.5, 0.95])
      print("Intensity Quantiles:", intensity_quantiles)
      print("Regularity Quantiles:", regularity_quantiles)
      print("Total Time Spent Quantiles:", total_time_spent_quantiles)
      print("Mean Intensity:", df['intensity'].mean())
      print("Mean Regularity:", df['regularity'].mean())
      print("Mean Total Time Spent:", df['total_time_spent'].mean())
      
      # 对user的行为进行分类
      # Define the thresholds for splitting the clusters
      regularity_threshold = 2
      intensity_threshold = 23

      # Create a new column 'cluster' based on the thresholds
      df['cluster'] = 0
      df.loc[(df['regularity'] > regularity_threshold) & (df['intensity'] > intensity_threshold), 'cluster'] = 1
      df.loc[(df['regularity'] > regularity_threshold) & (df['intensity'] <= intensity_threshold), 'cluster'] = 2
      df.loc[(df['regularity'] <= regularity_threshold) & (df['intensity'] > intensity_threshold), 'cluster'] = 3
      df.loc[(df['regularity'] <= regularity_threshold) & (df['intensity'] <= intensity_threshold), 'cluster'] = 4

      # Print the unique clusters found
      clusters = df['cluster'].unique()
      print("Clusters found:", clusters)
      print(df)
      # Print the number of students in each cluster
      cluster_counts = df['cluster'].value_counts()
      print("Number of students in each cluster:")
      for cluster, count in cluster_counts.items():
          print(f"Cluster {cluster}: {count} students")
      # Extract the cluster information into (user, cluster) pair
      behavior_clusters = pd.DataFrame(df, columns=['user_id', 'cluster'])
      
      # 对学生的成绩进行分类
      ## ToDo：
      # 这里不应该使用全部的成绩，而是应该用考试的成绩来做分类
      level = self.get_user_level_sequence(self.get_valid_user_sequence())
      ## ToDo：
      # 对level做一个normalization （把成绩从0-200变成0-100）
      # 之后在在0-100这个区间根据90，80，70，60来划分出A，B，C，D，F的等级
      level['cluster_y'] = 0
      level.loc[level['level'] > 150, 'cluster_y'] = 1
      level.loc[(level['level'] > 100) & (level['level'] <= 150), 'cluster_y'] = 2
      level.loc[(level['level'] > 50) & (level['level'] <= 100), 'cluster_y'] = 3
      level.loc[level['level'] <= 50, 'cluster_y'] = 4

      ## ToDo：
      # 根据新分好的成绩和行为的cluster来做一个sankey diagram表示出它们之间的关系
      # Extract the level cluster information into (user, cluster_y) pair
      level_cluster = level[['user_id', 'cluster_y']]
      # Count the number of each cluster
      cluster_counts = level_cluster['cluster_y'].value_counts()
      print("Number of users in each level cluster:")
      for cluster, count in cluster_counts.items():
          print(f"Cluster {cluster}: {count} users")
      print(level_cluster)
      
      # Merge the behavior clusters and level clusters
      merged_clusters = pd.merge(behavior_clusters, level_cluster, on='user_id')

      # Count the transitions between behavior clusters and level clusters
      transition_counts = merged_clusters.groupby(['cluster', 'cluster_y']).size().reset_index(name='count')

      # Create the Sankey diagram
      sankey = Sankey(unit=None)
      for _, row in transition_counts.iterrows():
          sankey.add(flows=[row['count'], -row['count']],
             labels=[f"Behavior Cluster {row['cluster']}", f"Level Cluster {row['cluster_y']}"],
             orientations=[0, 0])
      sankey.finish()
      plt.title("Sankey Diagram for Behavior Cluster and Level Cluster")
      plt.show()
      # Define the labels for the Sankey diagram
      labels = [f"Behavior Cluster {i}" for i in range(1, 5)] + [f"Level Cluster {i}" for i in range(1, 5)]

      # Define the source and target indices for the Sankey diagram
      source = []
      target = []
      value = []
      for _, row in transition_counts.iterrows():
        source.append(row['cluster'] - 1)
        target.append(row['cluster_y'] + 3)
        value.append(row['count'])

      # Create the Sankey diagram
      fig = go.Figure(data=[go.Sankey(
        node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
        ),
        link=dict(
        source=source,
        target=target,
        value=value
        ))])

      fig.update_layout(title_text="Sankey Diagram for Behavior Cluster and Level Cluster", font_size=10)
      fig.show()

      ## ToDo:
      # 把shengrui总结的那个word文档里的visualization都在这里计算掉
      # 1. intensity和regularity分布（cluster encode到color），cluster用的是行为的还是成绩的可以确认一下
      # 2. matrix图，行为的cluster和成绩的cluster的关系
      # 3. 不同的成绩里，四种behavior的分布柱状图
      # 4. 不同的behavior里，四种成绩的分布柱状图
      
    ######################################################################
    ######################################################################
    ############################# Overhead ###############################
    ######################################################################
    ######################################################################
      
    def get_time_score(self):
      user = util.load_json(util, "data/users.json")
      # get the time spent vs score for each question completed by the user
      time_spent_vs_score = []
      for i in range(len(user)):
        if not user[i].get("assessments"):
          continue
        for assessment in user[i]["assessments"]:
          time_spent_vs_score.append([assessment["duration"]/1000, assessment["score"]])
          
      # sort the time spent and scores
      time_spent_vs_score.sort(key=lambda x: x[0])
      df = pd.DataFrame(time_spent_vs_score, columns=["time_spent", "score"])
      return df
    
    def find_time_threshold(self):
      df = self.get_time_score()
      time_interval = [2, 3]
      step = 0.5
      result = []
      while time_interval[1] <= 93.5:
        interval_score = df[(df["time_spent"] >= time_interval[0]) & (df["time_spent"] < time_interval[1])]["score"]
        result.append([time_interval[1], interval_score.mean()])
        time_interval[1] += step
      # plot the result
      result = np.array(result)
      plt.plot(result[:, 0], result[:, 1])
      plt.xlabel("Time Spent")
      plt.ylabel("Score")
      plt.title("Score vs Time Spent")
      plt.show()
      
      quantiles = np.arange(0.05, 0.5, 0.01)
      quantile_values = [df["time_spent"].quantile(q) for q in quantiles]
      plt.figure(figsize=(12, 6))
      plt.subplot(1, 2, 1)
      plt.plot(quantiles, quantile_values, marker='o')
      plt.xlabel("Quantile")
      plt.ylabel("Time Spent")
      plt.title("Time Spent Lower Quantile")
      
      upper_quantile = 0.95
      upper_quantile_value = df["time_spent"].quantile(upper_quantile)
      print(upper_quantile_value)
      
      # plot the score for each quantile
      scores = [df[(df["time_spent"] >= q_val) & (df["time_spent"] <= upper_quantile_value)]["score"].mean() for q_val in quantile_values]
      plt.subplot(1, 2, 2)
      plt.plot(quantile_values, scores, marker='o')
      plt.xlabel("Time Spent")
      plt.ylabel("Score")
      plt.title("Score vs Time Spent Lower Quantile")
      plt.tight_layout()
      plt.show()
      
    def get_valid_time_score(self):
      df = self.get_time_score()
      lower_quantile_value = 2
      upper_quantile_value = 93.5
      return df[(df["time_spent"] <= upper_quantile_value) & (df["time_spent"] >= lower_quantile_value)]
    
    def time_score_kde_analysis(self):
      df = self.get_valid_time_score()
      print(df["time_spent"].mean(), df["time_spent"].std())
      sns.kdeplot(df["time_spent"], cmap="Blues", fill=True)
      plt.xlabel("Time Spent")
      plt.title("Distribution for Time Spent")
      plt.show()
    
    def get_valid_user_sequence(self):
      users = util.load_json(util, "data/users.json")
      valid_users = []
      for user in users:
        if not user.get("assessments"):
          continue
        valid_assessments = []
        for assessment in user["assessments"]:
          if assessment.get("duration") < 2000 or assessment.get("duration") > 93500:
            continue
          valid_assessments.append(assessment)
        if valid_assessments:
          user["assessments"] = valid_assessments
          valid_users.append(user)
      return valid_users
    
    def get_valid_user_exam_sequence(self):
      users = util.load_json(util, "data/users.json")
      valid_users = []
      for user in users:
        if not user.get("assessments"):
          continue
        valid_assessments = []
        for assessment in user["assessments"]:
          if assessment.get("duration") < 2000 or assessment.get("duration") > 93500:
            continue
          elif assessment.get("assessment_type") == 1 or assessment.get("assessment_type") == 8 or assessment.get("assessment_type") == 9:
            continue
          valid_assessments.append(assessment)
        if valid_assessments:
          user["assessments"] = valid_assessments
          valid_users.append(user)
      return valid_users
    
    def get_valid_user_exercise_sequence(self):
      users = util.load_json(util, "data/users.json")
      valid_users = []
      for user in users:
        if not user.get("assessments"):
          continue
        valid_assessments = []
        for assessment in user["assessments"]:
          if assessment.get("duration") < 2000 or assessment.get("duration") > 93500:
            continue
          elif assessment.get("assessment_type") == 5 or assessment.get("assessment_type") == 7:
            continue
          valid_assessments.append(assessment)
        if valid_assessments:
          user["assessments"] = valid_assessments
          valid_users.append(user)
      return valid_users
    
    def get_user_level_sequence(self, users):
      sequence = []
      question_info = util.load_json(util, "data/assessment_question_info.json")
      difficulty_mapping = {1: 1, 2: 1.2, 3: 2}
      for user in users:
        # sum the score of each assessment
        total_score = 0
        for assessment in user["assessments"]:
          total_score += assessment["score"] * difficulty_mapping[question_info[assessment["question"]]["difficulty"]]
        sequence.append([user["user_id"], total_score / len(user["assessments"])])
      
      return pd.DataFrame(sequence, columns=["user_id", "level"])
    
    def get_user_intensity_and_regularity_sequence(self, users):
      sequence = []
      for user in users:
        current_start_times = []
        for assessment in user["assessments"]:
          current_start_times.append(assessment["start_time"])
        current_start_times = [datetime.fromtimestamp(time) for time in current_start_times]
        week_numbers = [time.isocalendar()[1] for time in current_start_times]
        week_counts = {}
        for week in week_numbers:
          if week not in week_counts:
            week_counts[week] = 0
          week_counts[week] += 1
        intensity = sum(week_counts.values()) / len(week_counts)
        regularity = len(week_counts)
        sequence.append([user["user_id"], intensity, regularity])
      return pd.DataFrame(sequence, columns=["user_id", "intensity", "regularity"])

    
    def user_level_analysis(self, unit=-1):
      users = self.get_valid_user_sequence()
      valid_users = []
      for user in users:
        current_assessments = []
        for assessment in user["assessments"]:
          if not assessment:
            continue
          elif unit != -1 and unit not in assessment["unit"]:
            continue
          current_assessments.append(assessment)
        if current_assessments and len(current_assessments) > 20:
          user["assessments"] = current_assessments
          valid_users.append(user)
      sequence = self.get_user_level_sequence(valid_users)
      sns.kdeplot(sequence["level"], cmap="Blues", fill=True)
      plt.xlabel("Level")
      plt.show()
      print(sequence["level"].mean(), sequence["level"].std())
      print(sequence["level"].quantile(0.25), sequence["level"].quantile(0.75))
      print(sequence["level"].min(), sequence["level"].max())
      print(len(sequence))
    
    
    ######################################################################
    ######################################################################
    ########################### Markov Training ##########################
    ######################################################################
    ######################################################################
    
    # Convert question sequence to list of matrix that could be used as input to train Markov model directly.
    def convert_question_sequence_to_matrix(self, users, question_ids):
      matrix_sequence = []
      for user in users:
        assessment_matrix = np.zeros((len(user["assessments"]), len(question_ids)))
        tracking_vector = np.zeros(len(question_ids))
        for i, assessment in enumerate(user["assessments"]):
          question_index = question_ids.index(assessment["question"])
          tracking_vector[question_index] = 1 if tracking_vector[question_index] == 0 else 0
          assessment_matrix[i] = tracking_vector
        matrix_sequence.append(assessment_matrix)
      return matrix_sequence
      
    
    def kmeans_clustering(self):
      # user unit exercise data (not exam)
      users = self.get_valid_user_sequence()
      # get the level, intensity and regularity of each user and use kmeans to cluster the users
      level_sequence = self.get_user_level_sequence(users)
      intensity_regularity_sequence = self.get_user_intensity_and_regularity_sequence(users)
      # merge the two sequences
      merged_sequence = pd.merge(level_sequence, intensity_regularity_sequence, on="user_id")
      # normalize the data
      normalized_df = merged_sequence.copy()
      for column in ['level', 'intensity', 'regularity']:
        mean = merged_sequence[column].mean()
        std = merged_sequence[column].std()
        normalized_df[column] = (merged_sequence[column] - mean) / std
      # use kmeans to cluster the users
      kmeans = KMeans(n_clusters=3, random_state=0)
      clusters = kmeans.fit_predict(normalized_df[['level', 'intensity', 'regularity']])
      # add the cluster labels to the original data
      merged_sequence['cluster'] = clusters
      # visualize the clusters
      # sns.scatterplot(x='level', y='intensity', hue='cluster', data=merged_sequence, palette='viridis')
      # plt.title('User Clusters based on Level and Intensity')
      # plt.show()
      return merged_sequence
    
    def training_set(self):
      # only deal with the recommendation for unit 3
      assessment_questions_3 = util.load_json(util, "data/assessment_questions/12.json")
      users = self.get_valid_user_exercise_sequence()
      # filter only assessment questions that could be found in assessment_questions_3
      valid_users = []
      for user in users:
        current_assessments = []
        for assessment in user["assessments"]:
          if not assessment:
            continue
          elif assessment["assessment_id"] not in assessment_questions_3:
            print(assessment["assessment_id"], "not in assessment_questions_3")
            continue
          current_assessments.append(assessment)
        if current_assessments:
          user["assessments"] = current_assessments
          valid_users.append(user)
      cluster = self.kmeans_clustering()
      # Merge the valid users with their cluster information
      valid_users_df = pd.DataFrame(valid_users)
      print(valid_users_df)
      # valid_users_clustered = pd.merge(valid_users_df, cluster[['user_id', 'cluster']], on='user_id')

      # # Split the users based on their clusters
      # cluster_0_users = valid_users_clustered[valid_users_clustered['cluster'] == 0]
      # cluster_1_users = valid_users_clustered[valid_users_clustered['cluster'] == 1]
      # cluster_2_users = valid_users_clustered[valid_users_clustered['cluster'] == 2]

      # # Convert the question sequences to matrices for each cluster
      # question_ids = assessment_questions_3
      # cluster_0_matrix = self.convert_question_sequence_to_matrix(cluster_0_users.to_dict('records'), question_ids)
      # cluster_1_matrix = self.convert_question_sequence_to_matrix(cluster_1_users.to_dict('records'), question_ids)
      # cluster_2_matrix = self.convert_question_sequence_to_matrix(cluster_2_users.to_dict('records'), question_ids)

      # return [cluster_0_matrix, cluster_1_matrix, cluster_2_matrix]

if __name__ == "__main__":
    data_analysis = DataAnalysis()
    data_analysis.training_set()

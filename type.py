from typing import List
import numpy as np

class Assessment:
  def __init__(self, start_time: int, duration: int, score: int, question: str):
    self.start_time = start_time           # Start time of the assessment
    self.duration = duration               # Duration of the assessment
    self.score = score                     # Score of the assessment
    self.question = question               # List of Question objects

  def __repr__(self):
    return (f"Assessment(start_time={self.start_time}, duration={self.duration}, "
        f"score={self.score}, question={self.question})")


class User:
  def __init__(self, user_id: int, assessments: List[Assessment], questionIds):
    self.user_id = user_id                 # Unique user ID
    self.assessments = assessments      # List of Assessment objects
    self.assessments.sort(key=lambda x: x.start_time)
    self.question_Ids = questionIds
    self.assessment_matrix = self.get_matrix()

  def __repr__(self):
    return f"User(user_id={self.user_id}, assessments={self.assessments})"

  def get_matrix(self):
    # Convert the sorted order into a matrix using numpy
    assessment_matrix = np.zeros((len(self.assessments), len(self.question_Ids)))
    tracking_vector = np.zeros(len(self.question_Ids))
    for i, a in enumerate(self.assessments):
      question_index = self.question_Ids.index([a.question])
      tracking_vector[question_index] = 1 if tracking_vector[question_index] == 0 else 0
      assessment_matrix[i] = tracking_vector
    return assessment_matrix
  
  def get_next_question_id(self, encoding: List[int]):
    xor_result = np.logical_xor(encoding, self.assessment_matrix[-1])
    xor_result_indices = np.where(xor_result)[0]
    return self.question_Ids[xor_result_indices[0]]
  
  def get_score_vector(self):
    return [a.score for a in self.assessments]
  
  def get_duration_vector(self):
    return [a.duration for a in self.assessments]
  
  def get_start_time(self):
    return [a.start_time for a in self.assessments]
  
  def get_assessment_ids(self):
    return [a.question for a in self.assessments]
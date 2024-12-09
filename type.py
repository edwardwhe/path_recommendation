from typing import List
import numpy as np

class Assessment:
  def __init__(self, start_time: int, duration: int, score: int, question_id: str, assessment_id: str, assessment_type: int, unit: list[int]):
    self.start_time = start_time           # Start time of the assessment
    self.duration = duration               # Duration of the assessment
    self.score = score                     # Score of the assessment
    self.question_id = question_id         # List of Question objects
    self.assessment_id = assessment_id     # Belonged assessment ID
    self.assessment_type = assessment_type # Assessment type
    self.unit = unit                       # Unit of the assessment
    # __dict__ is used to convert the object into a dictionary
    self.__dict__ = {
        'start_time': self.start_time,
        'duration': self.duration,
        'score': self.score,
        'question': self.question_id,
        'assessment_id': self.assessment_id,
        'assessment_type': self.assessment_type,
        'unit': self.unit
    }

  def __repr__(self):
    return f"Assessment(start_time={self.start_time}, duration={self.duration}, score={self.score}, question_id={self.question_id}, assessment_id={self.assessment_id}, assessment_type={self.assessment_type})"


class User:
  def __init__(self, user_id: int, assessments: List[Assessment]):
    self.user_id = user_id                 # Unique user ID
    self.assessments = assessments      # List of Assessment objects
    self.assessments.sort(key=lambda x: x.start_time)
    self.__dict__ = {
        'user_id': self.user_id,
        'assessments': [a.__dict__ for a in self.assessments],
    }

  @classmethod
  def from_dict(cls, data):
    user_id = data['user_id']
    assessments = []
    for assessment_data in data['assessments']:
      start_time = assessment_data['start_time']
      duration = assessment_data['duration']
      score = assessment_data['score']
      question = assessment_data['question']
      question_id = assessment_data['assessment_id']
      assessment_type = assessment_data['assessment_type']
      unit = assessment_data['unit']
      assessment = Assessment(start_time, duration, score, question, question_id, assessment_type, unit)
      assessments.append(assessment)
    return cls(user_id, assessments)

  def __repr__(self):
    return f"User(user_id={self.user_id}, assessments={self.assessments})"
  
  def get_next_question_id(self, encoding: List[int], current_encoding: List[int]):
    xor_result = np.logical_xor(encoding, current_encoding)
    xor_result_indices = np.where(xor_result)[0]
    return xor_result_indices[0]
  
  
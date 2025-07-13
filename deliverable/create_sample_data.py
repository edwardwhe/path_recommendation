#!/usr/bin/env python3
"""
Create sample data for testing the Markov-based Educational Path Recommendation System.

This script generates minimal sample data to ensure the system can be tested even
if the full dataset is not available.
"""

import json
import random
import os
from datetime import datetime, timedelta

def create_sample_users_data(num_users=50, num_assessments_per_user=10):
    """Create sample users data."""
    print(f"Creating sample users data with {num_users} users...")
    
    # Sample question pool for unit 3
    unit_3_questions = [
        "question_3_1", "question_3_2", "question_3_3", "question_3_4", "question_3_5",
        "question_3_6", "question_3_7", "question_3_8", "question_3_9", "question_3_10",
        "question_3_11", "question_3_12", "question_3_13", "question_3_14", "question_3_15",
        "question_3_16", "question_3_17", "question_3_18", "question_3_19", "question_3_20"
    ]
    
    users = []
    base_time = int(datetime.now().timestamp()) - 86400 * 30  # 30 days ago
    
    for user_id in range(1, num_users + 1):
        assessments = []
        current_time = base_time + random.randint(0, 86400 * 7)  # Random start within first week
        
        for i in range(num_assessments_per_user):
            # Select random question
            question = random.choice(unit_3_questions)
            
            # Generate realistic assessment data
            assessment = {
                "start_time": current_time,
                "duration": random.randint(10000, 60000),  # 10-60 seconds
                "score": random.randint(60, 100),  # 60-100 score
                "question": question,
                "assessment_id": f"assess_{user_id}_{i}",
                "assessment_type": random.choice([1, 2, 3, 4]),  # Non-exam types
                "unit": [3]
            }
            assessments.append(assessment)
            
            # Increment time
            current_time += random.randint(3600, 86400)  # 1-24 hours later
        
        user = {
            "user_id": f"user_{user_id}",
            "assessments": assessments
        }
        users.append(user)
    
    return users

def create_sample_assessment_info():
    """Create sample assessment info."""
    print("Creating sample assessment info...")
    
    assessment_info = {}
    for user_id in range(1, 51):  # 50 users
        for i in range(10):  # 10 assessments per user
            assessment_id = f"assess_{user_id}_{i}"
            assessment_info[assessment_id] = {
                "type": random.choice([1, 2, 3, 4]),  # Non-exam types
                "syllabi": [3]  # Unit 3
            }
    
    return assessment_info

def create_sample_assessment_question_info():
    """Create sample assessment question info."""
    print("Creating sample assessment question info...")
    
    question_info = {}
    for i in range(1, 21):  # 20 questions
        question_id = f"question_3_{i}"
        question_info[question_id] = {
            "difficulty": random.choice([1, 2, 3]),  # Easy, Medium, Hard
            "grade": random.choice([6, 7, 8, 9, 10])  # Grade levels
        }
    
    return question_info

def create_sample_obj_to_unit():
    """Create sample objective to unit mapping."""
    print("Creating sample obj_to_unit mapping...")
    
    obj_to_unit = {}
    for i in range(1, 101):  # 100 objectives
        obj_id = f"obj_{i}"
        # Most objectives map to unit 3, some to other units
        if i <= 20:
            obj_to_unit[obj_id] = 3
        elif i <= 40:
            obj_to_unit[obj_id] = [3, 4]  # Some objectives span multiple units
        else:
            obj_to_unit[obj_id] = random.choice([1, 2, 3, 4, 5])
    
    return obj_to_unit

def create_sample_unit_questions():
    """Create sample unit questions."""
    print("Creating sample unit questions...")
    
    unit_questions = {}
    for unit in [1, 2, 3, 4, 5]:
        questions = [f"question_{unit}_{i}" for i in range(1, 21)]  # 20 questions per unit
        unit_questions[unit] = questions
    
    return unit_questions

def create_sample_user_ids():
    """Create sample user IDs."""
    return [f"user_{i}" for i in range(1, 51)]

def create_sample_assessment_question_ids():
    """Create sample assessment question IDs."""
    question_ids = []
    for unit in [1, 2, 3, 4, 5]:
        for i in range(1, 21):
            question_ids.append(f"question_{unit}_{i}")
    return question_ids

def ensure_data_directory():
    """Ensure data directory structure exists."""
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./data/assessment_questions"):
        os.makedirs("./data/assessment_questions")

def create_all_sample_data():
    """Create all sample data files."""
    print("Creating all sample data files...")
    
    ensure_data_directory()
    
    # Create sample data
    users_data = create_sample_users_data()
    assessment_info = create_sample_assessment_info()
    assessment_question_info = create_sample_assessment_question_info()
    obj_to_unit = create_sample_obj_to_unit()
    unit_questions = create_sample_unit_questions()
    user_ids = create_sample_user_ids()
    assessment_question_ids = create_sample_assessment_question_ids()
    
    # Save all data files
    files_to_create = [
        ("./data/users.json", users_data),
        ("./data/assessment_info.json", assessment_info),
        ("./data/assessment_question_info.json", assessment_question_info),
        ("./data/obj_to_unit.json", obj_to_unit),
        ("./data/user_id.json", user_ids),
        ("./data/assessment_question_id.json", assessment_question_ids)
    ]
    
    for file_path, data in files_to_create:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Created: {file_path}")
        else:
            print(f"Skipped (exists): {file_path}")
    
    # Create unit question files
    for unit, questions in unit_questions.items():
        file_path = f"./data/assessment_questions/{unit}.json"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(questions, f, indent=2)
            print(f"Created: {file_path}")
        else:
            print(f"Skipped (exists): {file_path}")

def check_existing_data():
    """Check what data files already exist."""
    print("Checking existing data files...")
    
    required_files = [
        "./data/users.json",
        "./data/assessment_info.json",
        "./data/assessment_question_info.json",
        "./data/obj_to_unit.json",
        "./data/user_id.json",
        "./data/assessment_question_id.json",
        "./data/assessment_questions/3.json"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            existing_files.append((file_path, file_size))
            print(f"✅ {file_path} ({file_size} bytes)")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} (missing)")
    
    print(f"\nSummary:")
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    return len(missing_files) == 0

def main():
    """Main function."""
    print("Sample Data Creator for Markov Educational Path Recommendation System")
    print("=" * 70)
    
    # Check existing data
    all_data_exists = check_existing_data()
    
    if all_data_exists:
        print("\n✅ All required data files already exist!")
        print("You can proceed with testing the system.")
    else:
        print("\n⚠️  Some data files are missing.")
        response = input("Would you like to create sample data files? (y/n): ").lower().strip()
        
        if response == 'y' or response == 'yes':
            create_all_sample_data()
            print("\n✅ Sample data creation completed!")
            print("You can now test the system with the sample data.")
        else:
            print("Sample data creation cancelled.")
    
    print("\nTo test the system, run:")
    print("python test_functions.py")

if __name__ == "__main__":
    main() 
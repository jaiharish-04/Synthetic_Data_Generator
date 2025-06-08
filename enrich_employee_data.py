import pandas as pd
import random
from faker import Faker
from datetime import datetime

# Load the original CSV with 50,000 employees
input_csv = "employee_dataset_50000.csv"
df = pd.read_csv(input_csv)

# Initialize Faker for Indian locale
fake = Faker('en_IN')

# Define constant lists
locations = ['Chennai', 'Bangalore', 'Mumbai', 'Coimbatore', 'Noida']
project_names = [
    'Apollo ERP', 'Zeus AI Ops', 'Hermes Portal', 'Athena BI', 'Orion Cloud Infra',
    'Poseidon DevOps', 'Helios CRM', 'HexaSecure', 'QuantumEdge', 'InsightX'
]
designations = [
    'Software Engineer', 'Data Analyst', 'Project Manager', 'DevOps Engineer',
    'System Architect', 'QA Tester', 'UI/UX Designer', 'Tech Lead', 'Scrum Master'
]

def random_phone():
    start_digit = random.choice(['6', '8', '9'])
    rest = ''.join([str(random.randint(0, 9)) for _ in range(9)])
    return f"+91 {start_digit}{rest}"


def random_email(emp_id):
    return f"{emp_id}@hexaware.com"

def random_location():
    return random.choice(locations)

def random_project():
    return random.choice(project_names)

def random_laptop():
    return random.choice([
        f"LTP-{random.randint(1000, 9999)}",
        f"LT-200{random.randint(10, 99)}"
    ])

def random_dob():
    dob = fake.date_of_birth(minimum_age=0, maximum_age=55)
    return dob.strftime("%d-%m-%Y")

def random_doj():
    start_date = datetime(2000, 1, 1)
    end_date = datetime.today()
    doj = fake.date_between(start_date=start_date, end_date=end_date)
    return doj.strftime("%d-%m-%Y")

def random_manager():
    return fake.name()

def random_designation():
    return random.choice(designations)

# Dictionary of field names and generators
field_generators = {
    'Phone Number': random_phone,
    'Email': None,  # handled separately
    'Location': random_location,
    'Project Name': random_project,
    'Laptop ID': random_laptop,
    'Date of Birth': random_dob,
    'Date of Joining': random_doj,
    'Manager Name': random_manager,
    'Designation': random_designation
}

# Process all records
output_data = []

for index, row in df.iterrows():
    emp_id = row['Employee ID']
    emp_name = row['Employee Name']
    
    # Pick 6 out of 9 fields randomly
    selected_fields = random.sample(list(field_generators.keys()), 6)

    # Basic record
    record = {
        'Employee Name': emp_name,
        'Employee ID': emp_id
    }

    # Add randomly selected fields
    for field in selected_fields:
        if field == 'Email':
            record[field] = random_email(emp_id)
        else:
            record[field] = field_generators[field]()

    output_data.append(record)

# Create and save new DataFrame
output_df = pd.DataFrame(output_data)
output_csv = "enriched_employee_dataset_50000.csv"
output_df.to_csv(output_csv, index=False)

print(f"File saved: {output_csv}")
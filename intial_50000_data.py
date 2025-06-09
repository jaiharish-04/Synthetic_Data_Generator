import pandas as pd
from faker import Faker
import random

# Load the first 10 employees from CSV
df_initial = pd.read_csv("employees.csv")  # Make sure employees.csv is in the same folder

# Setup Faker
fake = Faker("en_IN")
Faker.seed(0)

# Track used names and IDs
used_ids = set(df_initial["Employee ID"])
used_names = set(df_initial["Employee Name"])

# Function to generate a unique 10-digit employee ID starting with 2000
def generate_unique_employee_id():
    while True:
        emp_id = "2000" + str(random.randint(100000, 999999))
        if emp_id not in used_ids:
            used_ids.add(emp_id)
            return emp_id

# Function to generate a unique employee name
def generate_unique_name():
    while True:
        name = fake.name()
        if name not in used_names:
            used_names.add(name)
            return name

# Generate 49,990 additional unique employee records
new_employees = []
for _ in range(50000 - len(df_initial)):
    new_employees.append({
        "Employee Name": generate_unique_name(),
        "Employee ID": generate_unique_employee_id()
    })

# Create DataFrame from new records
df_new = pd.DataFrame(new_employees)

# Combine initial and new records
df_full = pd.concat([df_initial, df_new], ignore_index=True)

# Save to a new CSV file
df_full.to_csv("50000_employees.csv", index=False)

print("✅ File '50000_employees.csv' created with 50,000 unique employees.")
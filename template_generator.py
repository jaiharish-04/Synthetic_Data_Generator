import os
import json
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="d4f12df80419fff9",
    azure_endpoint="https://mavericks-secureapi.azurewebsites.net/api/azureai",
    api_version="2024-02-01"
)

# Predefined fields with a sample reference block for each (few-shot prompting style)
field_prompts = {
    "Phone Number": """Generate 10 professionally worded variations of the following question to collect an employee's official phone number.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your official phone number?
- Please enter the mobile number registered with the company.
- Share your work contact number.
- What phone number did you provide during onboarding?
- Kindly enter your full registered phone number.
- Type in the number where HR can contact you.
- What's the mobile number listed in your profile?
- Enter the 10-digit mobile number associated with your record.
- Provide the number linked to your Hexaware account.
- Give your complete registered mobile number.""",

    "Email": """Generate 10 professionally worded variations of the following question to collect an employee's Hexaware email address.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your official Hexaware email address?
- Enter your full work email ID.
- Which email address is linked to your employee record?
- Provide the email used for internal communication.
- Please type in your registered company email.
- What's your Hexaware email address?
- Give the complete email ID used during login.
- Mention your full corporate email address.
- Provide your email as per your employee profile.
- What is your work-assigned email ID?""",

    "Location": """Generate 10 professionally worded variations of the following question to collect an employee's current work location.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- Where is your current work location?
- In which city are you based?
- What is the official location assigned to you?
- Which branch/office are you working from?
- Mention your current city of operation.
- What city is registered as your base location?
- Please share your primary work city.
- What’s your office location?
- Enter the city where you are posted.
- Where is your workplace situated?""",

    "Project Name": """Generate 10 professionally worded variations of the following question to collect an employee's project name.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is the name of the project you’re working on?
- Please enter your current project title.
- Which project have you been assigned to?
- Mention your official project assignment.
- Type the project name listed in your records.
- Which project are you actively part of?
- Enter the name of your primary project.
- Provide the project name in your profile.
- Which team/project are you currently in?
- What's the project title associated with your work?""",

    "Laptop ID": """Generate 10 professionally worded variations of the following question to collect an employee's assigned laptop ID.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your official laptop ID?
- Enter the device number assigned to you.
- Share the serial ID of your Hexaware laptop.
- Provide your laptop’s identification code.
- What’s your issued system number?
- Give your registered laptop device ID.
- Mention your assigned laptop code.
- Please provide the laptop number on file.
- What device ID is linked to your profile?
- Type your assigned work system ID.""",

    "Date of Birth": """Generate 10 professionally worded variations of the following question to collect an employee's date of birth in YYYY-MM-DD format.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your full date of birth (YYYY-MM-DD)?
- Enter your date of birth as per records.
- Please provide your DOB in YYYY-MM-DD format.
- What is your date of birth?
- Mention your birthdate officially recorded.
- Give your birth date in full format.
- State your complete date of birth.
- Please enter your birth date exactly as provided.
- Provide your DOB in the required format.
- Kindly input your full date of birth.""",

    "Date of Joining": """Generate 10 professionally worded variations of the following question to collect an employee's date of joining.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your date of joining at Hexaware?
- When did you officially join the company?
- Please provide your joining date (YYYY-MM-DD).
- What is your onboarding date?
- Type the full date you joined the organization.
- Mention the start date of your employment.
- What’s your official joining date?
- Give the date when you became part of Hexaware.
- Enter your full joining date.
- Share the date of commencement of your role.""",

    "Manager Name": """Generate 10 professionally worded variations of the following question to collect an employee's reporting manager's name.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- Who is your reporting manager?
- Please provide your manager’s full name.
- Who do you report to?
- Mention the name of your assigned manager.
- What’s the full name of your supervisor?
- Provide the manager name from your record.
- Who is currently managing your work?
- Type in your manager’s name.
- Give the name of your immediate supervisor.
- Who is listed as your reporting authority?""",

    "Designation": """Generate 10 professionally worded variations of the following question to collect an employee's current job title.Do not include any explanations or introductory text. Only return the 10 questions as a numbered or bulleted list.:
- What is your current job title?
- Please provide your official designation.
- What is the role assigned to you?
- Mention your designation as per records.
- Give the title that appears in your employee profile.
- What position do you hold at Hexaware?
- Enter your work title.
- What’s your current role in the organization?
- Type your formal designation.
- Which role are you currently working in?"""
}

# Output dictionary
output_data = {}

# Generate questions for each field
for field, prompt in field_prompts.items():
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes HR-friendly questions for employee data collection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=700
        )

        # Extract and clean the response into a list
        questions_text = response.choices[0].message.content.strip()
        questions = [q.strip("-•1234567890. ").strip() for q in questions_text.split("\n") if q.strip()]
        output_data[field] = questions[:10]  # Ensure only 10

    except Exception as e:
        print(f"Error processing {field}: {str(e)}")

# Save to JSON
with open("templates_bank.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✅ JSON file 'templates_bank.json' created successfully.")

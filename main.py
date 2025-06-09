# === main.py ===
import json
from ml_selector import FieldTemplateSelector
from question_asker import QuestionAsker

# Load enriched employee data
with open("/Users/jaiharishsatheshkumar/synthetic_data_generator/.venv/bin/enriched_employee_dataset_50000.json") as f:
    user_data = json.load(f)

# Load logs
try:
    with open("logs.json") as f:
        logs = json.load(f)
except FileNotFoundError:
    logs = []

# Initialize the FieldTemplateSelector
selector = FieldTemplateSelector()

# Load existing logs into selector RL and logs list, then train ML
if logs:
    for log in logs:
        # Directly update RL Q-table without triggering repeated training
        reward = 1 if log.get('success', False) else -1
        selector.rl_selector.update_q(log['user_id'], log['field'], log['template'], reward)
        selector.logs.append(log)  # Append log without triggering training

    # Train ML model only once after loading all logs
    selector.train_supervised()

# Initialize question engine
asker = QuestionAsker(selector)

# === Step 1: Ask for employee ID ===
input_id = input("Enter your Employee ID: ").strip()

# === Step 2: Find employee record ===
record = next((r for r in user_data if str(r["Employee ID"]) == input_id), None)

if record:
    user_id = str(record["Employee ID"])
    user_name = record.get("Employee Name", "User")
    print(f"\n👋 Welcome, {user_name}!")

    # === Step 3: Ask 3 intelligent questions ===
    questions = asker.ask_questions(user_id, record, num_questions=3)
    print("\nPlease answer the following questions:\n")

    all_correct = True  # flag for overall result

    for field, template, question in questions:
        print(f"❓ {question}")
        user_answer = input("👉 Your answer: ").strip()
        correct_answer = record.get(field, "")
        # Normalize correct_answer to list for consistency
        correct_answers = [correct_answer] if not isinstance(correct_answer, list) else correct_answer
        asker.record_user_answer(user_id, field, template, user_answer, correct_answers)

        if user_answer.lower() not in [ans.lower() for ans in correct_answers]:
            all_correct = False

    # === Step 4: Final Access Decision ===
    if all_correct:
        print("\n✅ Success. You can pass through.")
    else:
        print("\n❌ Failure. Access denied.")

else:
    print("❌ Employee ID not found. Please check and try again.")

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier

class RLSelector:
    def __init__(self):
        self.q_table = defaultdict(float)
        self.recent_templates = defaultdict(lambda: deque(maxlen=3))

    def update_q(self, user_id, field, template, reward):
        key = (user_id, field, template)
        current_q = self.q_table.get(key, 0.0)
        learning_rate = 0.1
        discount_factor = 0.9
        new_q = current_q + learning_rate * (
            reward + discount_factor * self._max_future_q(user_id, field) - current_q
        )
        self.q_table[key] = new_q
        self.recent_templates[(user_id, field)].append(template)

    def _max_future_q(self, user_id, field):
        keys = [k for k in self.q_table if k[0] == user_id and k[1] == field]
        if not keys:
            return 0.0
        return max(self.q_table[k] for k in keys)

    def select_best(self, user_id, candidates, k=3):
        filtered_candidates = []
        for field, template in candidates:
            recent_for_field = self.recent_templates.get((user_id, field), deque())
            if template not in recent_for_field:
                filtered_candidates.append((field, template))

        if not filtered_candidates:
            for field, _ in candidates:
                self.recent_templates[(user_id, field)].clear()
            filtered_candidates = candidates

        filtered_candidates.sort(
            key=lambda ft: self.q_table.get((user_id, ft[0], ft[1]), 0.0), reverse=True
        )

        return filtered_candidates[:k]

class FieldTemplateSelector:
    def __init__(self):
        self.field_model = RandomForestClassifier()
        self.template_model = RandomForestClassifier()
        self.rl_selector = RLSelector()
        self.logs = []
        self.valid_users = set()
        self.cleaned_data = {}

        self.template_bank_path = "template_bank.json"
        self.template_bank = self._load_template_bank(self.template_bank_path)

        data_path = "/Users/jaiharishsatheshkumar/synthetic_data_generator/enriched_employee_dataset_50000.json"
        logs_path = "logs.json"

        self._load_valid_users(data_path)
        self._load_existing_logs(logs_path)
        self._load_cleaned_data(data_path)
        self.generate_small_logs(data_path, logs_path, 5000)

    def _load_template_bank(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load template bank from {path}: {e}")
            return {}

    def _load_valid_users(self, path):
        try:
            with open(path) as f:
                users = json.load(f)
                if isinstance(users, list):
                    self.valid_users = {str(user['Employee ID']) for user in users if 'Employee ID' in user}
        except Exception as e:
            print(f"Warning: Could not load valid users from {path}: {e}")

    def _load_existing_logs(self, path):
        if Path(path).exists():
            try:
                with open(path) as f:
                    self.logs = json.load(f)
                    for log in self.logs:
                        reward = 1 if log['success'] else -1
                        self.rl_selector.update_q(log['user_id'], log['field'], log['template'], reward)
            except Exception as e:
                print(f"Warning: Could not load logs from {path}: {e}")

    def _load_cleaned_data(self, path):
        try:
            with open(path) as f:
                records = json.load(f)
                for record in records:
                    user_id = str(record.get('Employee ID'))
                    self.cleaned_data[user_id] = {
                        k: [str(v)] if not isinstance(v, list) else [str(i) for i in v]
                        for k, v in record.items() if k != 'Employee ID'
                    }
        except Exception as e:
            print(f"Warning: Could not load cleaned data from {path}: {e}")

    def generate_small_logs(self, input_json_path, output_log_path, desired_log_count):
        try:
            with open(input_json_path) as f:
                records = json.load(f)

            logs = []
            random.seed(42)
            valid_logs = 0

            for user_record in records:
                if valid_logs >= desired_log_count:
                    break

                user_id = str(user_record.get('Employee ID', 'unknown'))

                fields = [
                    f for f in user_record
                    if f != 'Employee ID' and str(user_record[f]).strip() != "NA"
                ]
                if not fields:
                    continue

                field = random.choice(fields)
                correct_answer = str(user_record[field])
                template = f"Q-{field}-T{random.randint(1, 5)}"
                user_answer = correct_answer if random.random() < 0.8 else "invalid response"
                success = int(user_answer == correct_answer)

                logs.append({
                    "user_id": user_id,
                    "field": field,
                    "template": template,
                    "user_answer": user_answer,
                    "correct_answers": [correct_answer],
                    "success": success
                })

                valid_logs += 1

            with open(output_log_path, "w") as f:
                json.dump(logs, f, indent=2)

            print(f"Generated {len(logs)} logs and saved to {output_log_path}")

        except Exception as e:
            print(f"Error generating simulated logs: {e}")

    def validate_answer(self, user_id, field, user_answer):
        if len(self.logs) < 5:
            user_data = self.cleaned_data.get(user_id, {})
            return user_answer.lower() in [a.lower() for a in user_data.get(field, [])]

        try:
            X = [[int(user_id), hash(field) % 1000]]
            prediction = self.field_model.predict(X)[0]
            return bool(prediction)
        except Exception as e:
            print(f"ML validation error: {e}")
            return False

    def log_interaction(self, user_id, field, template, user_answer, correct_answers):
        success = self.validate_answer(user_id, field, user_answer)

        log = {
            "user_id": user_id,
            "field": field,
            "template": template,
            "user_answer": user_answer,
            "correct_answers": correct_answers,
            "success": success
        }
        self.logs.append(log)

        reward = 1 if success else -1
        self.rl_selector.update_q(user_id, field, template, reward)

        if len(self.logs) % 5 == 0:
            self.train_supervised()

        with open("logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)

    def train_supervised(self):
        if len(self.logs) < 5:
            return False
        try:
            X, y = [], []
            for log in self.logs:
                try:
                    uid = int(log['user_id'])
                    fid = hash(log['field']) % 1000
                    X.append([uid, fid])
                    y.append(log['success'])
                except:
                    continue
            if X and y:
                self.field_model.fit(np.array(X), np.array(y))
                print("ML model trained on", len(X), "examples")
                return True
            else:
                print("No data to train ML model.")
                return False
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def select_top_3_questions_for_random_field(self, user_id):
        if not self.is_valid_user(user_id):
            raise ValueError(f"Invalid user ID: {user_id}")

        if not self.template_bank:
            raise ValueError("Template bank is empty or not loaded.")

        selected_field = random.choice(list(self.template_bank.keys()))
        templates_for_field = self.template_bank[selected_field]
        candidates = [(selected_field, t) for t in templates_for_field]
        top_templates = self.rl_selector.select_best(user_id, candidates, k=3)
        return selected_field, [t for _, t in top_templates]

    def is_valid_user(self, user_id):
        return str(user_id) in self.valid_users


# Usage example:
if __name__ == "__main__":
    fts = FieldTemplateSelector()
    test_user_id = "12345"  # Replace with a valid user ID from your data

    try:
        field, questions = fts.select_top_3_questions_for_random_field(test_user_id)
        print("Selected Field:", field)
        print("Top 3 Questions:")
        for q in questions:
            print("-", q)
    except Exception as e:
        print("Error:", e)

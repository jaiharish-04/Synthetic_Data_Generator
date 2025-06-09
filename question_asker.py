import random
import json
import os
from collections import defaultdict, deque

class QuestionAsker:
    def __init__(self, selector, log_path="logs.json", history_path="question_history.json"):
        self.selector = selector
        self.cache = {}  # Cache templates per field
        self.log_path = log_path
        self.history_path = history_path
        self.asked_fields_by_user = self._load_user_history()
        self.recent_session_fields = defaultdict(lambda: deque(maxlen=3))  # session memory
        self.field_history = self._load_field_history()  # Persistent field-level history

    def _load_user_history(self):
        """
        Load persistent user-question logs to avoid long-term repetition.
        """
        asked = {}
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    logs = json.load(f)
                for entry in logs:
                    uid = entry['user_id']
                    field = entry['field']
                    if uid not in asked:
                        asked[uid] = set()
                    asked[uid].add(field)
            except Exception as e:
                print(f"Warning: Failed to load logs.json for history tracking: {e}")
        return asked

    def _load_field_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {self.history_path}: {e}")
        return {}

    def _save_field_history(self):
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.field_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save field history: {e}")

    def ask_questions(self, user_id, record, num_questions=3):
        """
        Ask up to `num_questions` from distinct fields randomly selected and not recently used.
        """
        # Step 1: Find eligible fields (excluding ID/Name, and ignoring "NA" values)
        populated_fields = [
            f for f in record
            if f not in ["Employee ID", "Employee Name"]
            and record[f] not in [None, "", [], "NA"]
        ]

        if not populated_fields:
            return []

        # Step 2: Load templates if not cached
        if not self.cache:
            try:
                self.cache = self.selector._load_template_bank("/Users/jaiharishsatheshkumar/synthetic_data_generator/.venv/bin/template_bank.json")
            except Exception as e:
                print(f"Warning: Could not load template bank: {e}")
                return []

        # Step 3: Keep only fields with templates
        valid_fields = [f for f in populated_fields if f in self.cache and self.cache[f]]
        if not valid_fields:
            return []

        # Step 4: Remove recently used fields from this session
        recent_fields = self.recent_session_fields[user_id]
        candidate_fields = [f for f in valid_fields if f not in recent_fields]

        if not candidate_fields:
            self.recent_session_fields[user_id].clear()
            candidate_fields = valid_fields

        # Step 5: Remove persistently asked fields from field history
        previously_asked = set(self.field_history.get(user_id, []))
        remaining_fields = [f for f in candidate_fields if f not in previously_asked]

        # If all fields already asked, reset persistent history for this user
        if not remaining_fields:
            self.field_history[user_id] = []
            remaining_fields = candidate_fields

        # Step 6: Shuffle and pick up to `num_questions` unique fields
        random.shuffle(remaining_fields)
        selected_fields = remaining_fields[:num_questions]

        questions = []

        for field in selected_fields:
            templates = self.cache.get(field, [])
            if not templates:
                continue
            candidates = [(field, t) for t in templates]
            best = self.selector.rl_selector.select_best(user_id, candidates, k=1)
            if best:
                selected_field, template = best[0]
                question_text = template.replace("{value}", str(record.get(selected_field, "")))
                questions.append((selected_field, template, question_text))
                self.recent_session_fields[user_id].append(field)

        # Step 7: Update persistent field history and save
        if user_id not in self.field_history:
            self.field_history[user_id] = []
        for field in selected_fields:
            if field not in self.field_history[user_id]:
                self.field_history[user_id].append(field)
        self._save_field_history()

        return questions

    def record_user_answer(self, user_id, field, template, user_answer, correct_answers):
        """
        Record user's answer, update RL scores, and avoid asking same field again.
        """
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]

        correct_answers = [ans.lower().strip() for ans in correct_answers]
        success = int(user_answer.lower().strip() in correct_answers)
        reward = 1 if success else -1

        # Update logs and Q-values
        self.selector.log_interaction(user_id, field, template, user_answer, correct_answers)
        self.selector.rl_selector.update_q(user_id, field, template, reward)

        # Track this field in permanent history (log-based only)
        if user_id not in self.asked_fields_by_user:
            self.asked_fields_by_user[user_id] = set()
        self.asked_fields_by_user[user_id].add(field)

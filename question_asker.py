import random

class QuestionAsker:
    def __init__(self, selector):
        """
        selector: instance of FieldTemplateSelector
        """
        self.selector = selector
        self.cache = {}  # Cache templates per field to avoid regenerating

    def ask_questions(self, user_id, record, num_questions=3):
        """
        Generate one question each from `num_questions` different fields.

        Returns a list of tuples: (field, template, question_text)
        """
        # Step 1: Get populated fields (excluding ID and name)
        populated_fields = [
            f for f in record 
            if f not in ["Employee ID", "Employee Name"] and record[f] not in [None, "", []]
        ]

        if not populated_fields:
            return []

        # Step 2: Load template bank once
        if not self.cache:
            self.cache = self.selector._load_template_bank(".venv/bin/template_bank.json")

        # Step 3: Shuffle populated fields to change variety
        random.shuffle(populated_fields)

        questions = []
        attempted_fields = set()

        for field in populated_fields:
            if len(questions) >= num_questions:
                break
            if field in attempted_fields:
                continue
            attempted_fields.add(field)

            templates = self.cache.get(field, [])
            if not templates:
                continue

            # Select best template using RL/ML
            candidate_templates = [(field, t) for t in templates]
            selected = self.selector.rl_selector.select_best(user_id, candidate_templates, k=1)

            if selected:
                field_selected, template = selected[0]
                question_text = template.replace("{value}", str(record.get(field_selected, "")))
                questions.append((field_selected, template, question_text))

        return questions

    def record_user_answer(self, user_id, field, template, user_answer, correct_answers):
        """
        Record the user's answer and update logs and RL Q-values.
        """
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]

        correct_answers = [ans.lower().strip() for ans in correct_answers]
        success = int(user_answer.lower().strip() in correct_answers)
        reward = 1 if success else -1

        self.selector.log_interaction(user_id, field, template, user_answer, correct_answers)
        self.selector.rl_selector.update_q(user_id, field, template, reward)

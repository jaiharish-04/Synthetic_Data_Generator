from collections import defaultdict, deque

class RLSelector:
    def __init__(self):
        # Q-table stores Q-values for (user_id, field, template)
        self.q_table = defaultdict(float)
        # Keep track of recently used templates per (user_id, field)
        self.recent_templates = defaultdict(lambda: deque(maxlen=3))

    def update_q(self, user_id, field, template, reward):
        key = (user_id, field, template)
        current_q = self.q_table.get(key, 0.0)
        learning_rate = 0.1
        discount_factor = 0.9
        # Q-learning update rule
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
        # Avoid recently used templates for user-field
        for field, template in candidates:
            recent_for_field = self.recent_templates.get((user_id, field), deque())
            if template not in recent_for_field:
                filtered_candidates.append((field, template))

        if not filtered_candidates:
            # If all templates are exhausted, reset recent templates
            for field, _ in candidates:
                self.recent_templates[(user_id, field)].clear()
            filtered_candidates = candidates

        # Sort by highest Q-value to pick best templates
        filtered_candidates.sort(
            key=lambda ft: self.q_table.get((user_id, ft[0], ft[1]), 0.0),
            reverse=True
        )

        return filtered_candidates[:k]

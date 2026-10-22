from locust import HttpUser, task, between
import random

SPAM_TEXTS = [
    "CLICK HERE to win FREE money now!!!",
    "Congratulations! You won $1000000 prize",
    "Buy cheap medications online, best prices",
    "LIMITED OFFER: Get rich quick scheme today",
    "FREE iPhone winner selected click now",
]

HAM_TEXTS = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the report by Friday?",
    "The meeting has been rescheduled to 3pm",
    "Thanks for your help with the project",
    "Are you coming to the team dinner tonight?",
]

class SpamClassifierUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_spam(self):
        self.client.post("/predict", json={
            "text": random.choice(SPAM_TEXTS)
        })

    @task(1)
    def predict_ham(self):
        self.client.post("/predict", json={
            "text": random.choice(HAM_TEXTS)
        })

    @task(1)
    def health_check(self):
        self.client.get("/")
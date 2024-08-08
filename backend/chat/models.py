from django.db import models

class Conversation(models.Model):
    user_input = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

class Schedule(models.Model):
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
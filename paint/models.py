from django.db import models
class Pic(models.Model):
    name=models.CharField(max_length=100)
    data=models.CharField(max_length=10000000000)

from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class UserProfile(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE,)
	email=models.EmailField()
	password=models.CharField(max_length=30)
	mobile=models.CharField(max_length=10)
	color=models.CharField(max_length=20)



	def __str__(self):
		return str(self.user.username)
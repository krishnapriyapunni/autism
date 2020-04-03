from django.shortcuts import render
from .models import *
from .views import *
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound,JsonResponse
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout


import sys
import scipy
import numpy
import matplotlib
import sklearn

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC




# Create your views here.
def home(request):
	return render(request,'index.html',{})

def signout(request):
	logout(request)
	return HttpResponseRedirect('/')

def log_in(request):
	if request.method == 'POST':
		email = request.POST.get('email')
		password = request.POST.get('password')
		user = authenticate(username=email,password=password)
		if user:
			print (user)
			login(request,user)
			return render(request,'index1.html', {"msg3":'Login Successfully'})
		else:
			# error="sorry"
			return render(request,'login.html',{"msg2":'UNABLE TO LOGIN '})
	else:
		return render(request,'login.html',{})
	# else:
	#   return render(request,'login.html',{})  
	return render(request,'index.html',{})


def signup(request):
	if request.method=='POST':
		name = request.POST.get('name')
		mobile = request.POST.get('mobile')
		password = request.POST.get('password')
		email = request.POST.get('email')
		color = request.POST.get('color')

		user1=UserProfile.objects.filter(email=email,password=password).exists()
		if not user1:   
			user2=User.objects.create_user(
				username=email,
				password=password,
				)
			
			user_pro=UserProfile.objects.create(
				user=user2,
				email=email,
				password=password,
				mobile=mobile,
				color=color,
			)
			user_pro.save()
			return render(request,'login.html',{"msg1":'you are logined'})
		else:
			error='you r already signed'
			return render(request,'signup.html',{'error':error})


		# return render(request,'login.html',{"msg1":'you are logined'})
	else:
		return render(request,'signup.html',{})
	return render(request,'signup.html',{})

def prediction(request):
# Load dataset
	dataset = pandas.read_csv("/home/ubuntu/projects/Documents/autism/mysitee/autism/data.csv/")
	# shape
	# Load dataset
# dataset = pandas.read_csv("x.csv")
# shape
	print(dataset.shape)
	# head
	print(dataset.head(20))
	# descriptions
	print(dataset.describe())
	# class distribution
	print(dataset.groupby('Class/ASD Traits ').size())

	# box and whisker plots
	#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	#plt.show()

	# histograms
	dataset.hist()
	plt.show()

	# scatter plot matrix
	scatter_matrix(dataset)
	plt.show()

	# Split-out validation dataset
	array = dataset.values
	X = array[:,0:12]
	print(X)
	Y = array[:,13]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'
	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
	models.append(('NB', GaussianNB()))
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)
	    
	# Compare Algorithms
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()    

	# Make predictions on validation dataset
	gn = GaussianNB()
	gn.fit(X_train, Y_train)
	predictions = gn.predict(X_validation)
	a=accuracy_score(Y_validation, predictions)
	b=confusion_matrix(Y_validation, predictions)
	c=classification_report(Y_validation, predictions)
		# plt.show()    
	
	return render(request,'prediction.html',{'a':a,'b':b,'c':c})
def about_us(request):
	return render(request,'about.html')

def contact(request):
	return render(request,'contact.html')

def result(request):
	return render(request,'result.html')

def asd(request):
	return render(request,'asd.html')

# def forgot(request):
# def forgot(request):

#     if request.method == 'POST':
#         email=request.POST.get('email')
#         color=request.POST.get('color')
#         # staffpassword=request.POST.get('staffpassword')
# 		# user22=Form.objects.filter(app_id=app_id)
# 		# user1 = authenticate(app_id=app_id)
#         # dep=Student.objects.get(course=department)
#         # dep1=dep.course
#         idd11=UserProfile.objects.get(email=email,color=color)
#         user444=idd11.email
#         user77=idd11.color
#         if user444 and user77:
# 			# print (idd1)
# 			# if dep:
# 				# views1=Form.objects.filter(department=dep.department)
# 				# views1=Student.objects.filter(department=dep.department).order_by('name')
# 			# views=Form.objects.all()
# 			# form = StudentForm()
#             posts=UserProfile.objects.filter(password=password)
# 			# view1=views.all()
#             return render(request,'forgot.html', {'posts':posts})
#             # if name:
#             #  	post1=Student.objects.filter(name=name)
#             #     return render(request,'listdetails.html', {'post1':post1})
# 			# else:
# 			# 	return render(request,'gallery.html',{})
#         else:
# 			# error="sorry"
#             return render(request,'forgot.html',{})
#     else:
#         return render(request,'forgot.html',{})
# def forgot(request):
# 	if request.method == 'POST':
# 		email=request.POST.get('email')
# 		color=request.POST.get('color')


# 		idd1=UserProfile.objects.get(email=email,color=color)
# 		email1=idd1.email
# 		color1=idd1.color
# 		if email1 and color1:
# 			# pas=UserProfile.objects.get(password=password)
# 			# print("pas",pas)
# 			return render(request,'forgot1.html', {'msg1':" its matches"})
# 		else:
# 			return render(request,'forgot.html',{})
# 	else:
# 		return render(request,'forgot.html',{})

# 	return render(request,'forgot.html',{})



# def forgot1(request):
# 	asd=UserProfile.objects.all()
# 	print("asd",asd)
# 	return render(request,'forgot1.html',{'asd':asd})

	# return render(request,'forgot.html')
def forgot(request):

	if request.method == 'POST':
		email = request.POST.get('email')
		# user22=Form.objects.filter(app_id=app_id)
		# user1 = authenticate(app_id=app_id)

		idd1=UserProfile.objects.get(email=email)
		user44=idd1.email
		if user44:
			# print (idd1)

			views=UserProfile.objects.filter(email=user44)
			# views=Form.objects.all()

			# view1=views.all()
			return render(request,'forgot1.html', {'views':views})
		else:
			# error="sorry"
			return render(request,'forgot.html',{})
	else:
		return render(request,'forgot.html',{})


def forgot1(request):
	return render(request,'forgot1.html')
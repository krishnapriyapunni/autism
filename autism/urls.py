from django.urls import path

from . import views


urlpatterns = [
	path('',views.home,name='home_url'),
	path('prediction/',views.prediction,name='prediction_url'),
	path('about_us/',views.about_us,name='about_us_url'),
	path('contact/',views.contact,name='contact_url'),
	path('result/',views.result,name='result'),
	path('signup/',views.signup,name='signup_url'),
	path('login/',views.log_in,name='login_url'),
	path('asd/',views.asd,name='asd_url'),
	path('forgot/',views.forgot,name='forgot_url'),
	path('forgot1/',views.forgot1,name='forgot1_url'),


	
]
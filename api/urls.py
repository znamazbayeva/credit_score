from django.urls import path
from .views import PredictCreditScore

urlpatterns = [
    path('predict/', PredictCreditScore.as_view(), name='predict_credit_score'),
]

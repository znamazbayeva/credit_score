from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
import joblib
import pandas as pd
import os
import io
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'best_svm_model.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))

with open(os.path.join(BASE_DIR, 'feature_names.pkl'), 'rb') as f:
    feature_names = joblib.load(f)

class PredictCreditScore(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                'file', openapi.IN_FORM, description="CSV file", type=openapi.TYPE_FILE, required=True
            ),
        ],
        responses={200: openapi.Response('Predictions', schema=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'predictions': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_STRING))
            }
        ))}
    )
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['file']
        df = pd.read_csv(io.StringIO(file_obj.read().decode('utf-8')))
        
        df = df[feature_names]
        
        predictions = model.predict(df)
        
        return Response({'predictions': predictions.tolist()}, status=status.HTTP_200_OK)

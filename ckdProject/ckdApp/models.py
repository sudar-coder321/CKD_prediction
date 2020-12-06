from django.db import models

# Create your models here.
class ckdModel(models.Model):

    Blood_Glucose_Random=models.FloatField()
    Blood_Urea=models.FloatField()
    Serum_Creatine=models.FloatField()
    Packed_cell_volume=models.FloatField()
    White_blood_count=models.FloatField()

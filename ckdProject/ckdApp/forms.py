from django import forms
from .models import *


class ckdForm(forms.ModelForm):
    class Meta():
        model=ckdModel
        fields=['Blood_Glucose_Random','Blood_Urea','Serum_Creatine','Packed_cell_volume','White_blood_count']

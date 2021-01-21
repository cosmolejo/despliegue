from django import forms
from .models import Pic


class ImageForm(forms.ModelForm):
    """Form for the image model"""

    class Meta:
        model = Pic
        fields = ['data','etnia']
        widgets = {'data': forms.HiddenInput()}

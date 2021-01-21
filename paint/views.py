from django.http.response import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect, render
from django.core.files import File

from django.template.loader import get_template
from django.template import RequestContext
from paint.models import Pic
from django.template import Context

from PIL import Image
import base64
from io import BytesIO
import numpy as np
from datetime import datetime
import pytz

from paint.face_generator import face_generator

face = face_generator()


def home(request):


    return render(request, "paint/paintapp.html")


def galle(request):
    return render(request, "paint/paintapp.html")


@csrf_exempt
def save(request):

    IST = pytz.timezone("America/Bogota")
    datetime_ist = datetime.now(IST)

    iname = str(datetime_ist.strftime("%Y_%m_%d_%H_%M"))
    irace = request.POST.get("race")
    idata = request.POST.get("data").split(",")[-1]

    byte_data = base64.b64decode(idata)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img = img.resize((256, 256))
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel

    # background.show()
    f_res = face.generate_face(img=background)
    f_res = f_res.resize((256, 256))

    f_res.show(title='Rostro generado')
   
    #data_url = "data:image/png;base64," + image_data

    p = Pic(name=iname, data=idata, etnia=irace)
    p.save()
    return render(request, "paint/paintapp.html", )


def gall(request):
    posts = [dict(id=i.id, title=i.name) for i in Pic.objects.order_by("id")]
    return render(request, "paint/gallery.html", {"posts": posts})


def load(request, imgname):
    data = Pic.objects.filter(name=imgname)
    print(data[0].id)
    for i in Pic.objects.filter(name=imgname):
        print(i.id)
    posts = [
        dict(id=i.id, title=i.name, imagedata=i.data)
        for i in Pic.objects.filter(name=imgname)
    ]
    return render(request, "picload.html", {"posts": posts})


# Create your views here.

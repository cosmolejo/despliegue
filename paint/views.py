from django.http.response import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect, render
from django.core.files import File
from django.conf import settings

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

import os


from paint.face_generator import face_generator

from .forms import ImageForm


face = face_generator()
CURR_DIR = os.getcwd()


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
    f_res = face.generate_face(background)
    f_res = f_res.resize((256, 256))

    f_res.show(title="Rostro generado")

    # data_url = "data:image/png;base64," + image_data

    p = Pic(name=iname, data=idata, etnia=irace)
    p.save()
    return render(
        request,
        "paint/paintapp.html",
    )


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


def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            img = Image.open(CURR_DIR + img_obj.image.url)
            img = img.resize((256, 256))
            f_res = face.generate_face(img)
            f_res = f_res.resize((256, 256))

            # f_res.show(title="Rostro generado")
            face_name = img_obj.image.url.split("/")[-1]
            a = os.path.split(os.getcwd())[:-1][0]
            print(a)
            f_res.save(CURR_DIR + "/paint/static/img/" + face_name)

            print(img_obj, img_obj.image.url)
            return render(
                request,
                "paint/carga.html",
                {"form": form, "img_obj": img_obj, "face": face_name},
            )
    else:
        form = ImageForm()
    return render(request, "paint/carga.html", {"form": form})

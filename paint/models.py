from django.db import models


class Pic(models.Model):

    etnias = [
        ("white", "Caucásico"),
        ("afro", "Afrodecendiente"),
        ("asian", "Asiatico"),
        ("arab", "Arabe"),
    ]
    name = models.CharField(max_length=100)
    etnia = models.CharField(
        max_length=100,
        choices=etnias,
        default='white',
    )
    data = models.CharField(max_length=10000000000)

    def __str__(self):
        return self.title


class Image(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.title
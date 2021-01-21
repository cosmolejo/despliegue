# Generated by Django 3.1.4 on 2021-01-21 02:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('paint', '0003_remove_pic_image'),
    ]

    operations = [
        migrations.CreateModel(
            name='image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('image', models.ImageField(upload_to='images')),
            ],
        ),
    ]

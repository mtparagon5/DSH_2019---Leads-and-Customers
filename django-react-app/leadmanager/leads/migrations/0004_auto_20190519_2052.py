# Generated by Django 2.2.1 on 2019-05-20 00:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('leads', '0003_auto_20190519_2036'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lead',
            name='score',
            field=models.FloatField(),
        ),
    ]
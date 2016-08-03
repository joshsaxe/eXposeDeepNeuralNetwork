#!/usr/bin/python

from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase

def makedb(dbfile):
    db = SqliteExtDatabase(dbfile)
    db.connect()

    global BaseModel
    class BaseModel(Model):
        class Meta:
            database = db

    global UniqueString
    class UniqueString(BaseModel):
        raw = CharField(index=True)
        count = IntegerField()
        malware_count = IntegerField()
        p_malware = FloatField(index=True)

    global Entity
    class Entity(BaseModel):
        name = CharField(index=True)
        label = FloatField()

    global String
    class String(BaseModel):
        uniquestring = ForeignKeyField(UniqueString,index=True)
        entity = ForeignKeyField(Entity,index=True)
        source = CharField(index=True,null=True)

    # create database
    db.create_tables([Entity,String,UniqueString])

    return db

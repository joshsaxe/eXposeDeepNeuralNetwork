import json
import sys
import argparse
import pprint
import datetime
import time

def daystrings(starttime,days):
    timestr = lambda t: datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%dT00:00:00')
    curtime = starttime
    strings = []
    for i in range(days):
        curtime+=86400
        day = timestr(curtime).split("T")[0]
        strings.append(day)
    return strings

"""
Example client for model_server.py
"""
import zerorpc
import pprint

c = zerorpc.Client()
c.connect("tcp://127.0.0.1:4242")
pprint.pprint( c.score(
    ["facebook.com",
    "paypal.com-confirm-your-paypal-account.spectragroup-inc.com/",
    "paypal.com-confirm-your-paypal-account.josh_made_this_up.com/",
    "paypal.com"]
) )

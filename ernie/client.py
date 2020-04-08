#coding=utf-8
import requests
import json
import time

start_time = time.time()
texta = "now suppose you have such a system"
textb = "such thin"
json_data = {'line':[texta, textb]}
r = requests.post('http://10.255.124.15:8808/prob_paddle', json = json_data)
print(json.loads(r.text)['prob'])
print("--- %s seconds ---" % (time.time() - start_time))


import matplotlib.pyplot as plt
import re
import json

txt=open("monitor.json").read()

ts=txt.split("\n")[1:] #skip header

dat=[]
for entry in ts:
    if len(entry)==0:
        break
    dat.append(json.loads(entry)["r"])

plt.plot(dat)
plt.savefig("log.png")

print("Output to log.png")



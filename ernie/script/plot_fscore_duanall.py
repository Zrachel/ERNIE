import matplotlib.pyplot as plt
import os
import sys
import pdb

def plot_all(dir, id, fileout):
    x = []
    y = []
    stepstart = 20000
    for i in range(1000000):
        step = stepstart + 2000 * i
        #print step
        filename=dir + "/" + "step_" + str(step)
        if os.path.exists(filename):
            with open(filename, 'r') as fin:
                res = [line.strip() for line in fin.readlines()]
            x.append(step)
            y.append(float(res[id]))
        else:
            continue
    #pdb.set_trace()
    plt.plot(x, y, 'go-', linewidth=2)
    plt.savefig(fileout)
    plt.close()

#dir="checkpoints_5cls_segment_withRegen_translation"
dir="checkpoints_cls2_LEN40_APPEND5_SEGMENT_force_context"
plot_all("log/"+dir, 1, "duanall.png")
plot_all("log/"+dir, 2, "clsall.png")
plot_all("log/"+dir, 3, "clsbiaodian.png")


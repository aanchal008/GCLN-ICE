import os
import os
 
# This is my path
path="traces/"
 
traces = []
 
# dirs=directories
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.csv' in f:
            traces.append(f)

for trace in traces:
    trace_file = open(path + trace, 'r')
    final_trace_file = open("traces_with_targets/" + trace, 'w')
    lines = trace_file.readlines()
    labels = lines[0].rstrip("\n")
    no_init_final = False
    if "init,final," not in labels:
        labels = "init,final," + labels
        no_init_final = True
        print(trace," doesn't have init final\n")
    labels = labels + ",target\n"

    final_trace_file.write(labels)
    for line in lines[1:-1]:
        init_final = ""
        if no_init_final:
            init_final = "0,0,"
        final_trace_file.write(init_final + line.rstrip("\n") + ",1\n")
    trace_file.close()
    final_trace_file.close()
    print("Updated to - " , "traces_with_targets/" + trace)
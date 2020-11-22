from pylsl import StreamInlet, resolve_stream
import time
from collections import deque

print(time.time())
start = time.time()

streams = resolve_stream("type", "EEG")

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

data = {}

fps = deque(maxlen=100)

channel = 8

while True:

    for i in range(channel):

        sample, timestamp = inlet.pull_sample()
        print(sample)
        # print(timestamp)
        if i not in data:
            data[i] = sample
        else:
            data[i].append(sample)

    fps.append(time.time() - start)
    # print("FPS: ", 1.0 / (time.time() - start))
    start = time.time()

    # print(1/(sum(fps)/len(fps)))

    # print(data)
    print("end of iteration \n")

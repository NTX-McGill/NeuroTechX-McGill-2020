from pylsl import StreamInlet, resolve_stream
import time
from collections import deque
import socketio

sio = socketio.Client()

print(time.time())
start = time.time()

stream_fft = resolve_stream("type", "FFT")

stream_series = resolve_stream("type", "Series")

# create a new inlet to read from the stream
inlet_fft = StreamInlet(stream_fft[0])
inlet_series = StreamInlet(stream_series[0])

sio.connect('http://localhost:5000')

@sio.event
def connect():
    print("Connected")

@sio.event
def message(data):
    print("Received Message")

for i in range(1000):

    print("FFT:")
    sample, timestamp = inlet_fft.pull_sample()
    print(sample)
    print(timestamp)
    sio.emit('fft', {'data': sample, 'timestamp': timestamp})

    print("Series:")
    sample, timestamp = inlet_series.pull_sample()
    print(sample)
    print(timestamp)
    sio.emit('series', {'data': sample, 'timestamp': timestamp})
    print("End Iteration")

    time.sleep(1)



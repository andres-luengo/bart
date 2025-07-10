from threading import Thread
from queue import Queue

def save_thread(data_queue: Queue):
    last_item = None
    while True:
        last_item = data_queue.get()

while True:
    
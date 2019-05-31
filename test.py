import numpy as np
import pandas as pd
import math
from pydub import AudioSegment
from pydub.utils import make_chunks
from datetime import datetime, timedelta

def main():

    #dt = datetime.now()

    #s = pd.to_datetime(dt)

    #s = s.timestamp()
    #print(date(s))



    myaudio = AudioSegment.from_file("hammering_lee2.wav", "wav")
    chunk_length_ms = 2000  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of two sec

    # Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")
        print('Exporting', chunk_name)


def date(dtordinal):
    days = dtordinal % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    py_datetime = datetime.fromordinal(int(dtordinal)) \
                  + timedelta(days=int(days)) \
                  + timedelta(hours=int(hours)) \
                  + timedelta(minutes=int(minutes)) \
                  + timedelta(seconds=round(seconds)) \
                  - timedelta(days=366)
    return py_datetime.date()


def time(dtordinal):
    days = dtordinal % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    py_datetime = datetime.fromordinal(int(dtordinal)) \
                  + timedelta(days=int(days)) \
                  + timedelta(hours=int(hours)) \
                  + timedelta(minutes=int(minutes)) \
                  + timedelta(seconds=round(seconds)) \
                  - timedelta(days=366)
    return py_datetime.time()

if __name__ == '__main__': main()
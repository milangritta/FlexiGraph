# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import sqlite3
import sys
from geopy.distance import great_circle
from keras.models import load_model
from subprocess import check_output
from preprocessing import print_stats, index_to_coord, generate_labels_from_file
from preprocessing import BATCH_SIZE, REVERSE_MAP_2x2
from preprocessing import generate_arrays_from_file

# For command line use, type: python test.py <dataset name>
if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    data = u"ShortDev"

weights_file = u"../data/weights"
print(u"Testing:", data, u"with weights:", weights_file)
word_to_index = cPickle.load(open(u"data/w2i.pkl"))
#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(weights_file)
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Crunching numbers, sit tight...')
# errors = codecs.open(u"errors.tsv", u"w", encoding=u"utf-8")  # Uncomment for diagnostics, also the section below.
conn = sqlite3.connect(u'../data/new.geonames.db')
file_name = u"../data/WikiGeolocate" + data + u".txt"
final_errors = []
for p, y in zip(model.predict_generator(generate_arrays_from_file(file_name, word_to_index, train=False),
                                 steps=int(check_output([u"wc", file_name]).split()[0]) / BATCH_SIZE, verbose=True),
                                 generate_labels_from_file(file_name)):
    p = index_to_coord(REVERSE_MAP_2x2[np.argmax(p)], 2)

    err = great_circle(p, y).km
    final_errors.append(err)

print_stats(final_errors)
print(u"Processed file", file_name)

# ---------------- DIAGNOSTICS --------------------
# import matplotlib.pyplot as plt
# plt.plot(range(len(choice)), np.log(1 + np.asarray(sorted(choice))))
# plt.xlabel(u"Predictions")
# plt.ylabel(u'Error Size')
# plt.title(u"Some Chart")
# plt.savefig(u'test.png', transparent=True)
# plt.show()
# ------------- END OF DIAGNOSTICS -----------------

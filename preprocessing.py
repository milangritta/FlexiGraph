# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
import matplotlib.pyplot as plt
import spacy, os
from lxml import etree
import numpy as np
import sqlite3
from geopy.distance import great_circle
from matplotlib import pyplot, colors


# -------- GLOBAL CONSTANTS AND VARIABLES -------- #
BATCH_SIZE = 64
CONTEXT_LENGTH = 2000
UNKNOWN = u"<unknown>"
EMBEDDING_DIMENSION = 50
ENCODING_MAP_1x1 = cPickle.load(open(u"data/1x1_encode_map.pkl"))      # We need these maps
ENCODING_MAP_2x2 = cPickle.load(open(u"data/2x2_encode_map.pkl"))      # and the reverse ones
REVERSE_MAP_1x1 = cPickle.load(open(u"data/1x1_reverse_map.pkl"))      # to handle the used and
REVERSE_MAP_2x2 = cPickle.load(open(u"data/2x2_reverse_map.pkl"))      # unused loc2vec polygons.
OUTLIERS_MAP_1x1 = cPickle.load(open(u"data/1x1_outliers_map.pkl"))    # Outliers are redundant polygons that
OUTLIERS_MAP_2x2 = cPickle.load(open(u"data/2x2_outliers_map.pkl"))    # have been removed but must also be handled.
# -------- GLOBAL CONSTANTS AND VARIABLES -------- #


def print_stats(accuracy):
    """
    Prints mean, median, AUC and acc@161km for the list.
    :param accuracy: a list of geocoding errors
    """
    print("==============================================================================================")
    print(u"Median error:", np.median(sorted(accuracy)))
    print(u"Mean error:", np.mean(accuracy))
    accuracy = np.log(np.array(accuracy) + 1)
    k = np.log(161)
    print u"Accuracy to 161 km: ", sum([1.0 for dist in accuracy if dist < k]) / len(accuracy)
    print u"AUC = ", np.trapz(accuracy) / (np.log(20039) * (len(accuracy) - 1))  # Trapezoidal rule.
    print("==============================================================================================")


def pad_list(size, a_list, from_left, padding):
    """
    Pads a given list with any given padding.
    :param size: the final length of the list i.e. pad up to size
    :param a_list: the list to pad
    :param from_left: True to pad from the left, False to pad from the right
    :param padding: whatever you want to use for padding, example "0"
    :return: the padded list
    """
    fill = [padding] * (size - len(a_list))
    if from_left:
        return fill + a_list
    else:
        return a_list + fill


def coord_to_index(coordinates, polygon_size):
    """
    Convert coordinates into an array index. Use that to modify loc2vec polygon value.
    :param coordinates: (latitude, longitude) to compute
    :param polygon_size: integer size of the polygon? i.e. the resolution of the world
    :return: index pointing into loc2vec array
    """
    latitude = float(coordinates[0]) - 90 if float(coordinates[0]) != -90 else -179.99  # The two edge cases must
    longitude = float(coordinates[1]) + 180 if float(coordinates[1]) != 180 else 359.99  # get handled differently!
    if longitude < 0:
        longitude = -longitude
    if latitude < 0:
        latitude = -latitude
    x = int(360 / polygon_size) * int(latitude / polygon_size)
    y = int(longitude / polygon_size)
    return x + y if 0 <= x + y <= int(360 / polygon_size) * int(180 / polygon_size) else Exception(u"Shock horror!!")


def index_to_coord(index, polygon_size):
    """
    Convert index to coordinates.
    :param index: of the polygon in loc2vec array
    :param polygon_size: size of each polygon i.e. resolution of the world
    :return: (latitude, longitude)
    """
    x = int(index / (360 / polygon_size))
    y = index % int(360 / polygon_size)
    if x > int(90 / polygon_size):
        x = -int((x - (90 / polygon_size)) * polygon_size)
    else:
        x = int(((90 / polygon_size) - x) * polygon_size)
    if y < int(180 / polygon_size):
        y = -int(((180 / polygon_size) - y) * polygon_size)
    else:
        y = int((y - (180 / polygon_size)) * polygon_size)
    return x, y


def get_coordinates(con, loc_name):
    """
    Access the database to retrieve coordinates and other data from DB.
    :param con: sqlite3 database cursor i.e. DB connection
    :param loc_name: name of the place
    :return: a list of tuples [(latitude, longitude, population, feature_code), ...]
    """
    result = con.execute(u"SELECT METADATA FROM GEO WHERE NAME = ?", (loc_name.lower(),)).fetchone()
    if result:
        result = eval(result[0])  # Do not remove the sorting, the function below assumes sorted results!
        return sorted(result, key=lambda (a, b, c): c, reverse=True)
    else:
        return []


def construct_loc2vec(a_list, polygon_size, mapping, outliers):
    """
    Build the loc2vec vector representation from a list of location data.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: what's the resolution? size of each polygon in degrees.
    :param mapping: one of the transformation maps 1x1 or 2x2
    :param outliers: the outlier map, 1x1 or 2x2 (must match resolution or mapping above)
    :return: loc2vec vector representation
    """
    loc2vec = np.zeros(len(mapping), )
    if len(a_list) == 0:
        return loc2vec
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        if index in mapping:
            index = mapping[index]
        else:
            index = mapping[outliers[index]]
        loc2vec[index] += float(max(s[2], 1)) / max_pop
    return loc2vec / loc2vec.max() if loc2vec.max() > 0.0 else loc2vec


def construct_loc2vec_full_scale(a_list, polygon_size):
    """
    This function is similar to the above BUT it builds loc2vec WITHOUT removing redundant polygons.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: size of each polygon in degrees i.e 1x1 or 2x2
    :return: loc2vec (full scale) i.e. without removing redundant polygons
    """
    loc2vec = np.zeros(int(360 / polygon_size) * int(180 / polygon_size))
    if len(a_list) == 0:
        return loc2vec
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        loc2vec[index] += float(max(s[2], 1)) / max_pop
    return loc2vec / loc2vec.max() if loc2vec.max() > 0.0 else loc2vec


def merge_lists(lists):
    """
    Utility function to merge multiple lists.
    :param lists: a list of lists to be merged
    :return: one single list with all items from above list of lists
    """
    out = []
    for l in lists:
        out.extend(l)
    return out


def populate_sql():
    """
    Create and populate the sqlite3 database with GeoNames data. Requires Geonames dump.
    No need to run this function, I share the database as a separate dump on GitHub (see link).
    """
    geo_names = {}
    p_map = {"PPLC": 100000, "PCLI": 100000, "PCL": 100000, "PCLS": 10000, "PCLF": 10000, "CONT": 100000, "RGN": 100000}

    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        feat_code = line[7]
        class_code = line[6]
        pop = int(line[14])
        for name in [line[1], line[2]] + line[3].split(","):
            name = name.lower()
            if len(name) != 0:
                if name in geo_names:
                    already_have_entry = False
                    for item in geo_names[name]:
                        if great_circle((float(line[4]), float(line[5])), (item[0], item[1])).km < 100:
                            if item[2] >= pop:
                                already_have_entry = True
                    if not already_have_entry:
                        pop = get_population(class_code, feat_code, p_map, pop)
                        geo_names[name].add((float(line[4]), float(line[5]), pop))
                else:
                    pop = get_population(class_code, feat_code, p_map, pop)
                    geo_names[name] = {(float(line[4]), float(line[5]), pop)}

    conn = sqlite3.connect(u'../data/new.geonames.db')
    c = conn.cursor()
    c.execute(u"CREATE TABLE GEO (NAME VARCHAR(100) PRIMARY KEY NOT NULL, METADATA VARCHAR(5000) NOT NULL);")
    # c.execute(u"DELETE FROM GEO")  # alternatively, delete the database file.
    # conn.commit()

    for gn in geo_names:
        c.execute(u"INSERT INTO GEO VALUES (?, ?)", (gn, str(list(geo_names[gn]))))
    print(u"Entries saved:", len(geo_names))
    conn.commit()
    conn.close()


def get_population(class_code, feat_code, p_map, pop):
    """
    Utility function to eliminate code duplication. Nothing of much interest, methinks.
    :param class_code: Geonames code for the class of location
    :param feat_code: Geonames code for the feature type of an database entry
    :param p_map: dictionary mapping feature codes to estimated population
    :param pop: population count
    :return: population (modified if class code is one of A, P or L.
    """
    if pop == 0 and class_code in ["A", "P", "L"]:
        pop = p_map.get(feat_code, 0)
    return pop


def generate_training_data(file_name):
    """
    Prepare Wikipedia training data. Please download the required files from GitHub.
    Files: geonames.db and geowiki.txt both inside the data folder.
    Alternatively, create your own with http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
    """
    conn = sqlite3.connect(u'../data/new.geonames.db')
    c = conn.cursor()
    if not os.path.exists("../data/"):
        os.mkdir("../data/")
    nlp = spacy.load(u'en')
    padding = nlp(u"0")[0]
    tree = etree.parse(u'WikiGeolocate/' + file_name + u".xml", etree.XMLParser(encoding='utf-8'))
    o = codecs.open(u"../data/" + file_name + ".txt", u"w", encoding=u"utf-8")

    for page in tree.getroot():
        lat = page.find("./latitude").text
        lon = page.find("./longitude").text
        text = unicode(page.text)
        locations, out = [], []
        doc = nlp(text)
        location = u""
        for index, item in enumerate(doc[:CONTEXT_LENGTH]):
            if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
                if item.ent_iob_ == u"B" and item.text.lower() == u"the":
                    out.append(item.text.lower())
                else:
                    location += item.text + u" "
                    out.append(item.text.lower())
            elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY",
                                    u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
                out.append(padding)
            elif item.is_punct:
                out.append(padding)
            elif item.is_digit or item.like_num:
                out.append(padding)
            elif item.like_email:
                out.append(padding)
            elif item.like_url:
                out.append(padding)
            elif item.is_stop:
                out.append(padding)
            else:
                out.append(item.lemma_)
            if location.strip() != u"" and item.ent_type == 0:
                location = location.strip()
                coords = get_coordinates(c, location)
                if len(coords) > 0:
                    locations.append(coords)
                location = u""
        entities = merge_lists(locations)
        o.write(lat + u"\t" + lon + u"\t" + str(out))
        o.write(u"\t" + str(entities) + u"\n")
    o.close()


def visualise_2D_grid(x, title, log=False):
    """
    Display 2D array data with a title. Optional: log for better visualisation of small values.
    :param x: 2D numpy array you want to visualise
    :param title: of the chart because it's nice to have one :-)
    :param log: True in order to log the values and make for better visualisation, False for raw numbers
    """
    if log:
        x = np.log10(x)
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap', ['lightgrey', 'darkgrey', 'dimgrey', 'black'])
    cmap.set_bad(color='white')
    img = pyplot.imshow(x, cmap=cmap, interpolation='nearest')
    pyplot.colorbar(img, cmap=cmap)
    plt.title(title)
    # plt.savefig(title + u".png", dpi=200, transparent=True)  # Uncomment to save to file
    plt.show()


def generate_vocabulary(path, min_words, min_entities):
    """
    Prepare the vocabulary for training/testing.
    :param path: to the file from which to build
    :param min_words: occurrence for inclusion in the vocabulary
    :param min_entities: occurrence for inclusion in the vocabulary
    """
    vocab_words = {UNKNOWN, u'0'}
    words = []
    for f in [path]:  # You can also build the vocabulary from several files, just add to the list.
        training_file = codecs.open(f, u"r", encoding=u"utf-8")
        for line in training_file:
            line = line.strip().split("\t")
            words.extend(eval(line[2]))

    words = Counter(words)
    for word in words:
        if words[word] > min_words:
            vocab_words.add(word)
    print(u"Words saved:", len(vocab_words))

    word_to_index = dict([(w, i) for i, w in enumerate(vocab_words)])
    cPickle.dump(word_to_index, open(u"data/w2i.pkl", "w"))


def generate_arrays_from_file(path, w2i, train=True):
    """
    Generator function for the FULL CNN + LOC2VEC model in the paper. Uses all available data inputs.
    :param path: to the training file (see training data generation functions)
    :param w2i: the vocabulary set
    :param train: True is generating training data, false for test data
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        words, labels = [], []
        loc2vec = []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_loc2vec([(float(line[0]), float(line[1]), 0)], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))
            words.append(pad_list(CONTEXT_LENGTH, eval(line[2]), from_left=False, padding=u'0'))

            # loc2vec.append(construct_loc2vec(sorted(eval(line[4]) + eval(line[6]) + eval(line[7]),
            #                key=lambda (a, b, c, d): c, reverse=True), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))
            loc2vec.append(construct_loc2vec(eval(line[3]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

            if counter % BATCH_SIZE == 0:
                for x in words:
                    for i, w in enumerate(x):
                        if w in w2i:
                            x[i] = w2i[w]
                        else:
                            x[i] = w2i[UNKNOWN]
                if train:
                    yield ([np.asarray(words), np.asarray(words), np.asarray(words), np.asarray(loc2vec)], np.asarray(labels))
                else:
                    yield ([np.asarray(words), np.asarray(words), np.asarray(words), np.asarray(loc2vec)])

                words, labels = [], []
                loc2vec = []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for x in words:
                for i, w in enumerate(x):
                    if w in w2i:
                        x[i] = w2i[w]
                    else:
                        x[i] = w2i[UNKNOWN]
            if train:
                yield ([np.asarray(words), np.asarray(words), np.asarray(words), np.asarray(loc2vec)], np.asarray(labels))
            else:
                yield ([np.asarray(words), np.asarray(words), np.asarray(words), np.asarray(loc2vec)])


def generate_strings_from_file(path):
    """
    Generator of labels, location names and context. Used for training and testing.
    :param path: to the training file (see training data generation functions)
    :return: Yields a list of tuples [(label, location name, context), ...]
    """
    while True:
        for line in codecs.open(path, "r", encoding="utf-8"):
            line = line.strip().split("\t")
            context = u" ".join(eval(line[2])) + u"*E*" + u" ".join(eval(line[5])) + u"*E*" + u" ".join(eval(line[3]))
            yield ((float(line[0]), float(line[1])), u" ".join(eval(line[5])).strip(), context)


def generate_arrays_from_file_loc2vec(path, train=True, looping=True):
    """
    Generator for the plain loc2vec model, works for MLP, Naive Bayes or Random Forest.
    :param path: to the training file (see training data generation functions)
    :param train: True for training phase, False for testing phase
    :param looping: True for continuous generation, False for one iteration.
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        labels, target_coord = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_loc2vec([(float(line[0]), float(line[1]), 0, u'')], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))
            target_coord.append(construct_loc2vec(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

            if counter % BATCH_SIZE == 0:
                if train:
                    yield ([np.asarray(target_coord)], np.asarray(labels))
                else:
                    yield ([np.asarray(target_coord)])

                labels = []
                target_coord = []

        if len(labels) > 0:
            # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            if train:
                yield ([np.asarray(target_coord)], np.asarray(labels))
            else:
                yield ([np.asarray(target_coord)])
        if not looping:
            break


def shrink_loc2vec(polygon_size):
    """
    Remove polygons that only cover oceans. Dumps a dictionary of DB entries.
    :param polygon_size: the size of each polygon such as 1x1 or 3x3 degrees (integer)
    """
    loc2vec = np.zeros((180 / polygon_size) * (360 / polygon_size),)
    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        lat, lon = float(line[4]), float(line[5])
        index = coord_to_index((lat, lon), polygon_size=polygon_size)
        loc2vec[index] += 1.0
    cPickle.dump(loc2vec, open(u"loc2vec.pkl", "w"))


def oracle(path):
    """
    Calculate the Oracle (best possible given your database) performance for a given dataset.
    Prints the Oracle scores including mean, media, AUC and acc@161.
    :param path: file path to evaluate
    """
    final_errors = []
    conn = sqlite3.connect(u'../data/new.geonames.db')
    for line in codecs.open(path, "r", encoding="utf-8"):
        line = line.strip().split("\t")
        coordinates = (float(line[0]), float(line[1]))
        best_candidate = []
        for candidate in get_coordinates(conn.cursor(), u" ".join(eval(line[5])).strip()):
            best_candidate.append(great_circle(coordinates, (float(candidate[0]), float(candidate[1]))).km)
        final_errors.append(sorted(best_candidate)[0])
    print_stats(final_errors)


# --------------------------------------------- INVOKE FUNCTIONS ---------------------------------------------------

# print get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"dublin")
# generate_training_data("WikiGeolocateShortDev")
# generate_evaluation_data(corpus="lgl", file_name="")
# generate_vocabulary(path=u"../data/train_wiki.txt", min_words=9, min_entities=1)
# shrink_loc2vec(2)
# oracle(u"data/eval_geovirus.txt")
# conn = sqlite3.connect('../data/new.geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"darfur", u"[(13.5, 23.5, 0), (44.05135, -94.83804, 106)]"))
# c.execute("DELETE FROM GEO WHERE name = 'darfur'")
# conn.commit()
# print index_to_coord(8177, 2)
# populate_sql()

# -------- CREATE MAPS (mapping from 64,000/16,200 polygons to 23,002, 7,821) ------------
# l2v = list(cPickle.load(open(u"data/new.geonames_1x1.pkl")))
# zeros = dict([(i, v) for i, v in enumerate(l2v) if v > 0])  # isolate the non zero values
# zeros = dict([(i, v) for i, v in enumerate(zeros)])         # replace counts with indices
# zeros = dict([(v, i) for (i, v) in zeros.iteritems()])      # reverse keys and values
# cPickle.dump(zeros, open(u"data/1x1_encode_map.pkl", "w"))

# ------- VISUALISE THE WHOLE DATABASE ----------
# l2v = np.reshape(l2v, newshape=((180 / 1), (360 / 1)))
# visualise_2D_grid(l2v, "Geonames Database", True)

# -------- CREATE OUTLIERS (polygons outside of loc2vec) MAP --------
# filtered = [i for i, v in enumerate(l2v) if v > 0]
# the_rest = [i for i, v in enumerate(l2v) if v == 0]
# poly_size = 2
# dict_rest = dict()
#
# for poly_rest in the_rest:
#     best_index = 100000
#     best_dist = 100000
#     for poly_filtered in filtered:
#         dist = great_circle(index_to_coord(poly_rest, poly_size), index_to_coord(poly_filtered, poly_size)).km
#         if dist < best_dist:
#             best_index = poly_filtered
#             best_dist = dist
#     dict_rest[poly_rest] = best_index
#
# cPickle.dump(dict_rest, open(u"data/2x2_outliers_map.pkl", "w"))

# ------ PROFILING SETUP -----------
# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()
# CODE HERE
# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()

# ----------- VISUALISATION OF DIFFERENT LOCATIONS -------------
# print len(get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"Melbourne"))
# coord = get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"Melbourne")
# coord.extend(get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"Victoria"))
# coord.extend(get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"Newcastle"))
# coord.extend(get_coordinates(sqlite3.connect('../data/new.geonames.db').cursor(), u"Perth"))
# coord = sorted(coord, key=lambda (a, b, c, d): c, reverse=True)
# x = construct_loc2vec_full_scale(coord, polygon_size=3)
# x = np.reshape(x, newshape=((180 / 3), (360 / 3)))
# visualise_2D_grid(x, "Melbourne", True)

# http://api.geonames.org/searchJSON?q=london&maxRows=10&inclBbox=true&username=milangritta

# ---------- DUMP DATABASE ------
# import sqlite3
#
# con = sqlite3.connect('../data/new.geonames.db')
# with codecs.open('dump.sql', 'w', 'utf-8') as f:
#     for line in con.iterdump():
#         f.write('%s\n' % line)
# -------------------------------

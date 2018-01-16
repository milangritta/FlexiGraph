import codecs
import sqlite3
from preprocessing import get_coordinates
from lxml import etree

inp = codecs.open(u"../data/geowiki.txt", u"r", encoding=u"utf-8")
string, counter = u"", 0
conn = sqlite3.connect('../data/new.geonames.db').cursor()
rootShort = etree.Element("WikiGeolocate")
rootLong = etree.Element("WikiGeolocate")

for line in inp:
    if len(line.strip()) == 0:
        continue
    if line.startswith(u"NEW ARTICLE::"):
        if 100 < len(string.split()) < 300:
            root = rootShort
        elif 300 < len(string.split()) < 2000:
            root = rootLong
        else:
            string = ""
        if len(string.strip()) > 0 and len(get_coordinates(conn, name)) > 0:
            page = etree.SubElement(root, "page", name=unicode(name))
            etree.SubElement(page, "latitude").text = lat
            etree.SubElement(page, "longitude").text = lon
            page.text = string
            counter += 1
        line = line.strip().split("\t")
        name = line[1]
        lat = line[2]
        lon = line[3]
        string = ""
    else:
        string += line
    if counter > 10000:
        break

for root in [(rootShort, "Short"), (rootLong, "Long")]:
    for t in [(0.8, "Train"), (0.5, "Dev"), (1, "Test")]:
        end = int(len(root[0]) * t[0])
        new = etree.Element("WikiGeolocate")
        for item in root[0][:end]:
            new.append(item)
        tree = etree.ElementTree(new)
        tree.write("WikiGeolocate/WikiGeolocate" + root[1] + t[1] + ".xml", pretty_print=True, xml_declaration=True, encoding="UTF-8")

# 693821
# [(0, 371822), (1, 97986), (2, 52695), (3, 40694), (4, 27060), (5, 21650), (6, 15348), (7, 11149), (8, 8473), (9, 6626), (10, 5335), (11, 4392), (12, 3594), (13, 3070), (14, 2481), (15, 2146), (16, 1875), (17, 1677), (18, 1374), (19, 1239), (20, 1074), (21, 955), (22, 816), (23, 802), (24, 676), (25, 602), (27, 557), (26, 530), (28, 432), (29, 427), (30, 369), (32, 341), (31, 338), (34, 290), (33, 270), (35, 267), (37, 230), (36, 210), (38, 204), (39, 192), (40, 176), (41, 155), (43, 145), (42, 143), (44, 136), (45, 128), (46, 127), (48, 110), (47, 106), (50, 104), (52, 104), (51, 96), (54, 86), (49, 83), (53, 78), (55, 73), (59, 71), (60, 64), (56, 63), (64, 54), (61, 52), (69, 51), (58, 50), (57, 49), (63, 49), (65, 48), (66, 47), (67, 43), (62, 38), (77, 38), (78, 38), (71, 35), (76, 34), (68, 32), (73, 32), (70, 29), (74, 28), (72, 26), (75, 26), (79, 26), (80, 25), (89, 24), (82, 23), (91, 22), (83, 21), (81, 20), (85, 20), (88, 20), (84, 19), (86, 18), (94, 18), (100, 18), (104, 18), (87, 16), (92, 16), (93, 16), (96, 16), (97, 16), (90, 15), (107, 14), (98, 13), (101, 13), (95, 11), (99, 10), (114, 10), (126, 10), (102, 9), (108, 9), (117, 9), (103, 8), (106, 8), (110, 8), (113, 8), (115, 8), (105, 7), (111, 7), (116, 7), (119, 7), (120, 7), (132, 7), (109, 6), (112, 6), (125, 6), (129, 5), (131, 5), (139, 5), (121, 4), (128, 4), (134, 4), (135, 4), (136, 4), (137, 4), (141, 4), (122, 3), (124, 3), (127, 3), (130, 3), (138, 3), (142, 3), (152, 3), (157, 3), (167, 3), (123, 2), (133, 2), (144, 2), (155, 2), (156, 2), (158, 2), (161, 2), (165, 2), (171, 2), (172, 2), (204, 2), (118, 1), (140, 1), (143, 1), (145, 1), (146, 1), (150, 1), (151, 1), (153, 1), (154, 1), (159, 1), (162, 1), (163, 1), (175, 1), (176, 1), (186, 1), (187, 1), (194, 1), (195, 1), (202, 1)]

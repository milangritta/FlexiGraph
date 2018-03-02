# coding=utf-8
import networkx as nx
import spacy
import matplotlib.pyplot as plt


nlp = spacy.load(u'en')
# text = u"The U.S. Food and Drug Administration (FDA) has issued a recall on salmonella " \
#        u"contaminated pistachios for 31 states in the United States. The salmonella contamination was " \
#        u"discovered by Kraft foods during routine testing last Tuesday, before any illness were reported. " \
#        u"Setton Farms based in California, the pistachio supplier, is voluntarily recalling their pistachios. " \
#        u"They notified the FDA and the FDA notified Setton Farms. Signs of salmonella include fever, diarrhea, " \
#        u"nausea, vomiting and abdominal pain. However, it is expected that the recalled list may grow as the " \
#        u"investigation continues. Kroger Co. is recalling shelled pistachios called Private Selection Shelled " \
#        u"Pistachios in a 10-ounce container with UPC code 111073615 and the sell dates of December 13 or 14 on the packages."

text = u'The agriculture officials earlier announced that the depopulation will be carried out in a "humane" manner, ' \
       u'following current Office international des épizooties (OIE) procedures that ensure protection of animal ' \
       u'welfare in the Bulacan farm. According to Philippine Department of Health (DOH) Secretary Francisco Duque, ' \
       u'the quarantine of the hog farm in Palauig, Manaoag, Pangasinan has been lifted after finding no traces of ' \
       u'viral transmission. According to Yap, test results conducted by a joint mission of FAO, the World Animal ' \
       u'Health Organization, WHO and their local counterparts, reveal that viral transmission continues to exist in ' \
       u'Pandi hog farms, which is only 0.5% of the 13 million pigs raised throughout the country. DOH officials also ' \
       u'say a pig farm worker in Cabanatuan in Nueva Ecija, who had no direct contact with sick hogs, has tested ' \
       u'positive for Immunoglobulin G antibodies against the Reston ebolavirus, which is non-lethal, unlike the ' \
       u'Zaïre, Bundibugyo, Côte d\'Ivoire and Sudan strains, according to FAO. '

graph = nx.Graph()
doc = nlp(text)
entities = dict()

for entity in doc.ents:
    if entity.label_ in [u"CARDINAL", u"DATE", u"PRODUCT"]:
        continue
    for index in range(entity.start, entity.end):
        ent_string = entity.text.lower()
        if ent_string.startswith(u"the"):
            ent_string = ent_string[3:].strip()
        entities[index] = ent_string
        head = entity.root.head
        if head.i in entities:
            graph.add_edge(ent_string, entities[head.i])
            continue
        while head.is_stop and head.dep_ != u"ROOT" or head.is_digit:
            head = head.head
        graph.add_edge(ent_string, head.text.lower())

for d in doc:
    head = d.head
    if d.is_stop or d.is_punct or head == d or d.is_digit or d.i in entities:
        continue
    if head.i in entities:
        graph.add_edge(d.text.lower(), entities[head.i])
        continue
    while head.is_stop and head.dep_ != u"ROOT" or head.is_digit:
        if head.i in entities:
            print head
        head = head.head
    graph.add_edge(d.text.lower(), head.text.lower(), weight=1)

nx.draw_spring(graph, with_labels=True, node_size=1000, node_shape='o', node_color='lightblue')
plt.show()

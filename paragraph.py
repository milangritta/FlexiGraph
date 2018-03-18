# coding=utf-8
import codecs
import networkx as nx
import numpy
import spacy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


nlp = spacy.load(u'en_core_web_lg')
text = codecs.open("../data/geovirus/57", encoding="utf-8").read()
graph = nx.Graph()


def sentences2edges(sentence_one, sentence_two, s_one_i, s_two_i):
    sentence_one = nlp(sentence_one)
    sentence_two = nlp(sentence_two)
    # Measure the interconnectedness of articles? Topical coherence?
    for entity_one in sentence_one.ents:
        if entity_one.label_ in [u"CARDINAL", u"DATE"]:
            continue
        for entity_two in sentence_two.ents:
            if entity_two.label_ in [u"CARDINAL", u"DATE"]:
                continue
            if entity_one.has_vector and entity_two.has_vector:
                if entity_one.similarity(entity_two) > 0.7:
                    graph.add_edge(s_one_i + u"-" + unicode(entity_one.root.i) + u"--" + entity_one.root.lower_,
                                   s_two_i + u"-" + unicode(entity_two.root.i) + u"--" + entity_two.root.lower_,
                                   weight=entity_one.similarity(entity_two), colour='b')
            elif entity_one.lower_ == entity_two.lower_:
                graph.add_edge(s_one_i + u"-" + unicode(entity_one.root.i) + u"--" + entity_one.root.lower_,
                               s_two_i + u"-" + unicode(entity_two.root.i) + u"--" + entity_two.root.lower_,
                               weight=1.0, colour='b')
    for chunk_one in sentence_one.noun_chunks:
        if len(chunk_one) == 1:
            continue
        for chunk_two in sentence_two.noun_chunks:
            if len(chunk_two) == 1:
                continue
            if chunk_one.has_vector and chunk_two.has_vector:
                if chunk_one.similarity(chunk_two) > 0.85:
                    u = s_one_i + u"-" + unicode(chunk_one.root.i) + u"--" + chunk_one.root.lower_
                    v = s_two_i + u"-" + unicode(chunk_two.root.i) + u"--" + chunk_two.root.lower_
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v, weight=chunk_one.similarity(chunk_two), colour='r')
            elif chunk_one.lower_ == chunk_two.lower_:
                u = s_one_i + u"-" + unicode(chunk_one.root.i) + u"--" + chunk_one.root.lower_
                v = s_two_i + u"-" + unicode(chunk_two.root.i) + u"--" + chunk_two.root.lower_
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, weight=1.0, colour='r')


sentences = list(nlp(text).sents)
node_colours = iter(cm.rainbow(numpy.linspace(0, 1, len(sentences))))
print text

for index, c in zip(range(len(sentences)), node_colours):
    sentence = sentences[index].text
    for word in nlp(sentences[index].text):
        graph.add_node(unicode(index) + u"-" + unicode(word.i) + u"--" + word.lower_, colour=c)
        if word.dep_ == u"ROOT":
            continue
        graph.add_edge(unicode(index) + u"-" + unicode(word.i) + u"--" + word.lower_,
                       unicode(index) + u"-" + unicode(word.head.i) + u"--" + word.head.lower_,
                       weight=1.0, colour='k')
    for sub_index in range(index + 1, len(sentences)):
        other_sentence = sentences[sub_index].text
        sentences2edges(sentence, other_sentence, unicode(index), unicode(sub_index))

#  TRY A BIGGER ARTICLE LIKE A WHOLE DOCUMENT, THESE MAY BE TOO SMALL
#  Could try both, boosting map2vec with flexiGRAPH in addition to default and using ONLY flexiGRAPH weights
colors = [graph[u][v]['colour'] for u, v in graph.edges()]
nodes = [data['colour'] for node, data in graph.nodes().items()]
weights = [graph[u][v]['weight'] * 3 if graph[u][v]['colour'] != 'k' else graph[u][v]['weight'] for u, v in graph.edges()]
# print nx.shortest_path(graph, u"0-Bulacan-37", u"4-Sudan-50", weight='weight')
nx.draw_kamada_kawai(graph, edgelist=graph.edges(), with_labels=True, node_shape='o', font_family='serif',
                     node_color=nodes, width=weights, edge_color=colors, node_size=200)
plt.show()

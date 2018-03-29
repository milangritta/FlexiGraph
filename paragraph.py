# coding=utf-8
import codecs
import networkx as nx
import numpy
import spacy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


nlp = spacy.load(u'en_core_web_lg')
text = codecs.open("../data/lgl/83", encoding="utf-8").read()
graph = nx.Graph()


def graph_to_sim_matrix(g, m):
    """
    Exponentiate the semantic distances for greater distinction?
    Ranks instead of Prior Probability?
    :param g:
    :param m:
    :return:
    """
    table = [['Similarity Table'] + m.keys()]
    for e1 in m:
        row = [e1]
        for e2 in m:
            if nx.has_path(g, m[e1], m[e2]):
                row.append(float("{0:.2f}".format(nx.shortest_path_length(g, m[e1], m[e2], weight='weight'))))
            else:
                row.append(numpy.inf)
        table.append(row)
    s = [[unicode(e) for e in row] for row in table]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = u'\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print u'\n'.join(table)


def build_graph(sentence_one, sentence_two, s_one_i, s_two_i):
    """

    :param sentence_one:
    :param sentence_two:
    :param s_one_i:
    :param s_two_i:
    :return:
    """
    sentence_one = nlp(sentence_one)
    sentence_two = nlp(sentence_two)
    # Measure the interconnectedness of articles? Topical coherence?
    for entity_one in sentence_one.ents:
        if entity_one.label_ in [u"CARDINAL", u"DATE", u"TIME", u"ORDINAL"]:
            continue
        for entity_two in sentence_two.ents:
            if entity_two.label_ in [u"CARDINAL", u"DATE", u"TIME", u"ORDINAL"]:
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
entity_map = dict()
print text

for index, c in zip(range(len(sentences)), iter(cm.rainbow(numpy.linspace(0, 1, len(sentences))))):
    sentence = sentences[index].text
    for entity in nlp(sentences[index].text).ents:
        print entity.text, entity.label_
        entity_map[entity.text] = unicode(index) + u"-" + unicode(entity.root.i) + u"--" + entity.root.lower_
    for word in nlp(sentences[index].text):
        is_entity = 'y' if word.ent_type != 0 else 'n'
        graph.add_node(unicode(index) + u"-" + unicode(word.i) + u"--" + word.lower_, colour=c, entity=is_entity)
        if word.dep_ == u"ROOT":
            continue
        graph.add_edge(unicode(index) + u"-" + unicode(word.i) + u"--" + word.lower_,
                       unicode(index) + u"-" + unicode(word.head.i) + u"--" + word.head.lower_,
                       weight=1.0, colour='k')
    for sub_index in range(index + 1, len(sentences)):
        other_sentence = sentences[sub_index].text
        build_graph(sentence, other_sentence, unicode(index), unicode(sub_index))


print graph_to_sim_matrix(graph, entity_map)
# nx.node_connectivity(graph) page rank? google_matrix?
# print sorted(nx.degree_centrality(graph).items(), key=lambda (x, y): y)
# TRY A BIGGER ARTICLE LIKE A WHOLE DOCUMENT, THESE MAY BE TOO SMALL
# DeLallo Italian & DeLallo - No similarity due to missing vector for "DeLallo".
# Solution? LGL/420 and Vincent J. Fumo and Fumo WIKI/407
# Could try both, boosting map2vec with flexiGRAPH in addition to default and using ONLY flexiGRAPH weights
# Levenstein distance? Pingliang city & Pingliang -- China & China Daily
colors = [graph[u][v]['colour'] for u, v in graph.edges()]
nodes = [data['colour'] for node, data in graph.nodes().items()]
widths = [2.5 if data['entity'] == 'y' else 1 for node, data in graph.nodes().items()]
lines = ['dashed' if graph[u][v]['colour'] != 'k' else 'solid' for u, v in graph.edges()]
weights = [graph[u][v]['weight'] * 3 if graph[u][v]['colour'] != 'k' else graph[u][v]['weight'] for u, v in graph.edges()]
nx.draw_kamada_kawai(graph, edgelist=graph.edges(), with_labels=True, node_shape='o', font_family='serif',
                     node_color=nodes, width=weights, edge_color=colors, node_size=200, linewidths=widths, style=lines)
plt.show()

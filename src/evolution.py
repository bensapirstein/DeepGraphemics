import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx
# from encoding import FontGraphemeAnalyzer

relationships = {
    "Hieroglyph": ["Proto-Sinaitic"],
    "Proto-Sinaitic": ["Ancient South-Arabian", "Ancient North-Arabian", "Phoenician"],
    "Ancient South-Arabian": ["Ge'ez"],
    "Phoenician": ["Paleo-Hebrew", "Aramaic"],
    "Paleo-Hebrew": ["Samaritan"],
    "Aramaic": ["Syriac", "Nabataean", "Hebrew", "Parthian", "Palmyrene", "Hatran", "Elymaic", "Pahlavi"],
    "Nabataean": ["Arabic"],
    "Syriac": ["Old Sogdian"],
}

letters = ['ʾalp', 'bayt', 'gaml', 'dalt', 'hillul', 'waw', 'zayn', 'ḥaṣr', 'ṭab', 'yad', 'kap', 'lamd', 'maym',
           'naḥaš', 'ṡamk', 'ʿayn', 'pu', 'ṣaday', 'ṣappa', 'qoba', 'raʾš', 'šimš', 'šin', 'tāw', 'ġabiʿ', 'ḫayt',
           'dag', 'ẓil', 'ṯad', 'ṯann']


# add fonts to the font manager
def add_fonts(analyzer):
    for font_path in analyzer.font_encoding["font_path"]:
        font_manager.fontManager.addfont(font_path)


# create a directed graph
def create_graph(relationships, graphemes, letter):
    G = nx.DiGraph()
    node_labels = {}

    for script, children in relationships.items():
        if not G.has_node(script):
            symbol = graphemes[graphemes["script"]==script][letter].values[0][0]
            node_labels[script] = f"{symbol}"
            G.add_node(script, symbol=symbol)

        for child in children:
            if not G.has_node(child):
                symbol = graphemes[graphemes["script"]==child][letter].values[0][0]
                node_labels[child] = f"{symbol}"
                G.add_node(child, symbol=symbol)

            G.add_edge(script, child)

    return G, node_labels

def topo_pos(G):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0.0
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)

    return pos_dict

# plot the graph with fonts
def plot_graph(G, node_labels, letter, font_paths, outpath=None):
    #pos = nx.spring_layout(G)
    pos = topo_pos(G)

    plt.figure(figsize=(12, 12))

    def draw_networkx_fonts(G, pos, labels, font_size=64):
        for node, label in labels.items():
            x, y = pos[node]
            font = font_manager.FontProperties(fname=font_paths[node])
            plt.text(x, y, label, fontproperties=font, fontsize=40, horizontalalignment="center", verticalalignment="center")

    draw_networkx_fonts(G, pos, node_labels)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, arrowstyle="-|>", connectionstyle="arc3,rad=0.1", alpha=0.5)
    plt.title(letter, fontdict={"family": "Arial", "size": 32})
    plt.axis("off")
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()

def plot_evolutionary_tree(letter, relationships, fga, fonts_dir, outpath=None):
    # extract scripts from relationships keys and values
    evo_scripts = list(relationships.keys()) + [child for children in relationships.values() for child in children]
    evo_scripts = list(set(evo_scripts))

    font_paths = {}
    selected_fonts = []
    for script in evo_scripts:
        font_file = fga.scripts.loc[script]["font path"]
        font_name = os.path.splitext(os.path.basename(font_file))[0]
        selected_fonts.append(fga.font_encoding.loc[font_name])
        font_path = os.path.join(fonts_dir, font_file)
        font_paths[script] = font_path

    graphemes = pd.concat(selected_fonts, axis=1).T

    G, node_labels = create_graph(relationships, graphemes, letter)
    plot_graph(G, node_labels, letter, font_paths, outpath=outpath)


def main():
    fga = FontGraphemeAnalyzer("../data/encoding/scripts.csv",
                               "../data/encoding/alphabet_allographic_etymology.csv",
                               "../data/encoding/alphabet_specific_encodings.csv",
                               "../data/encoding/font_encoding.csv",
                               "../data/fonts")

    # plot the evolution of the letter "aleph"
    plot_evolutionary_tree("aleph", relationships, fga, "../data/semiticRegular")

if __name__ == "__main__":
    main()

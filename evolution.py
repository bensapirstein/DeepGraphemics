import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx

# load scripts from csv file
def load_data():
    scripts = pd.read_csv("data/scripts.csv", index_col=0)
    graphemes = pd.read_csv("data/etymology_table.csv", index_col=0)
    return scripts, graphemes

# add fonts to the font manager
def add_fonts(font_paths):
    fonts_dir = "data/semiticRegular"
    for font_path in font_paths:
        font_manager.fontManager.addfont(os.path.join(fonts_dir, font_path))

# create a directed graph
def create_graph(relationships, graphemes, letter):
    G = nx.DiGraph()
    node_labels = {}

    for script, children in relationships.items():
        if not G.has_node(script):
            symbol = graphemes[script][letter]
            node_labels[script] = f"{symbol}"
            G.add_node(script, symbol=graphemes[script][letter])

        for child in children:
            if not G.has_node(child):
                symbol = graphemes[child][letter]
                node_labels[child] = f"{symbol}"
                G.add_node(child, symbol=graphemes[child][letter])

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
def plot_graph(G, node_labels, font_names, letter):
    pos = nx.spring_layout(G)
    pos = topo_pos(G)

    def draw_networkx_fonts(G, pos, labels, font_names, font_size=64):
        for node, label in labels.items():
            x, y = pos[node]
            font_family = font_names[node]
            plt.text(x, y, label, fontdict={"family": font_family, "size": font_size}, horizontalalignment="center", verticalalignment="center")

    draw_networkx_fonts(G, pos, node_labels, font_names)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, arrowstyle="-|>", connectionstyle="arc3,rad=0.1")
    plt.title(letter, fontdict={"family": "Arial", "size": 32})
    plt.axis("off")
    plt.show()

def main():
    scripts, graphemes = load_data()

    font_names = scripts["font name"]
    font_paths = scripts["font path"].tolist()
    add_fonts(font_paths)

    os.makedirs("output", exist_ok=True)

    for letter in graphemes.index:
        G, node_labels = create_graph(relationships, graphemes, letter)
        plot_graph(G, node_labels, font_names, letter)

if __name__ == "__main__":
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

    main()

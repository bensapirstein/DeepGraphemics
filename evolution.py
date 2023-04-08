import matplotlib.pyplot as plt
import math
import networkx as nx
import random
import os
import matplotlib.font_manager as font_manager
import pandas as pd

# load scripts from csv file
scripts = pd.read_csv("data/scripts.csv", index_col=0)
graphemes = pd.read_csv("data/etymology_table.csv", index_col=0)

# Define the relationships between scripts
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
alt_Alef_symbol = "ℵ"

# Define scripts with several styles
styles = {
    "Hebrew": {
        "Rashi": "Rashi",
        "Solitreo": "Solitreo",
    }
}

# print pwd
print(os.getcwd())

font_names = scripts["font name"]
print(font_names)

font_paths = scripts["font path"].tolist()
# print font files under semiticRegular directory

fonts_dir = "data/semiticRegular"
print(os.listdir(fonts_dir))

# add fonts to the font manager
for font_path in font_paths:
    font_manager.fontManager.addfont(os.path.join(fonts_dir, font_path))


ב = "bayt"
letter = ב

def create_graph(relationships, graphemes, letter):
    # Create a directed graph
    G = nx.DiGraph()
    node_labels = {}

    # Add edges
    for script, children in relationships.items():

        # If G doesn't have node for script, add it
        if not G.has_node(script):
            symbol = graphemes[script][letter]
            node_labels[script] = f"{symbol}"
            G.add_node(script, symbol=graphemes[script][letter])

        for child in children:
            # If G doesn't have node for child, add it
            if not G.has_node(child):
                symbol = graphemes[child][letter]
                node_labels[child] = f"{symbol}"
                G.add_node(child, symbol=graphemes[child][letter])

            G.add_edge(script, child)

    return G, node_labels

#pos = nx.planar_layout(G)
#pos = hierarchy_pos(G, "Hieroglyph")

def topo_pos(G):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0.0
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)

    return pos_dict

def plot_graph(G, node_labels, letter):

    pos = topo_pos(G)

    def draw_networkx_fonts(G, pos, labels, font_names, font_size=64):
        for node, label in labels.items():
            x, y = pos[node]
            font_family = font_names[node]
            plt.text(x, y, label, fontdict={"family": font_family, "size": font_size}, horizontalalignment="center", verticalalignment="center")

    # Draw labels with fonts
    draw_networkx_fonts(G, pos, node_labels, font_names)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, arrowstyle="-|>", connectionstyle="arc3,rad=0.1")
    #nx.draw_networkx(G, with_labels=True, labels=node_labels, node_shape="none", font_size=12, font_family="Arial")
    plt.title(letter, fontdict={"family": "Arial", "size": 32})

    # Show plot
    plt.axis("off")
    plt.show()
    # save plot
    #plt.savefig(f"output/{letter}.png", dpi=300)
    # clear plot
    #plt.clf()


plt.figure(3, figsize=(12, 8))
# make output directory
os.makedirs("output", exist_ok=True)

G, node_labels = create_graph(relationships, graphemes, graphemes.index[0])
plot_graph(G, node_labels, graphemes.index[0])

# for letter in graphemes.index:
#     G, node_labels = create_graph(relationships, graphemes, letter)
#
#     plot_graph(G, node_labels, letter)

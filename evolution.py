import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import networkx as nx
from encoding import FontGraphemeAnalyzer

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
            symbol = graphemes[graphemes["script"]==script][letter].values[0]
            node_labels[script] = f"{symbol}"
            G.add_node(script, symbol=symbol)

        for child in children:
            if not G.has_node(child):
                symbol = graphemes[graphemes["script"]==child][letter].values[0]
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

def plot_evolutionary_tree(letter, relationships, font_encoding, outpath=None):
    # extract scripts from relationships keys and values
    scripts = list(relationships.keys()) + [child for children in relationships.values() for child in children]
    scripts = list(set(scripts))

    selected_fonts = []
    for script in scripts:
        script_fonts = font_encoding[font_encoding["script"] == script]
        sample = script_fonts.sample(n=1, replace=True)
        selected_fonts.append(sample)
        font_manager.fontManager.addfont(sample["font_path"].values[0])

    graphemes = pd.concat(selected_fonts)
    font_paths = dict(zip(scripts, graphemes["font_path"].values))

    G, node_labels = create_graph(relationships, graphemes, letter)
    plot_graph(G, node_labels, letter, font_paths, outpath=outpath)


def main():
    analyzer = FontGraphemeAnalyzer()
    add_fonts(analyzer)

    os.makedirs("output", exist_ok=True)
    n_frames = 4
    frame_duration = 0.4  # in seconds
    letters = letters[:24]

    for letter in letters:
        # create a list of image file paths for the letter's evolution trees
        image_files = []
        for i in range(n_frames):
            outpath = f"output/{letter}_{i}.png"
            plot_evolutionary_tree(letter, relationships, analyzer.font_encoding, outpath=outpath)
            image_files.append(outpath)

        # concatenate the images into a video using ffmpeg
        video_name = f"output/{letter}.mp4"
        image_list = "|".join(image_files)
        os.system(f"ffmpeg -y -loglevel error -framerate 1/{frame_duration} -i 'concat:{image_list}' -c:v libx264 -vf 'fps=25,format=yuv420p' {video_name}")

        # delete the image files
        for file_path in image_files:
            os.remove(file_path)

    # concatenate the videos into a single video using ffmpeg
    video_files = [f"{letter}.mp4" for letter in letters]
    video_list = "\n".join([f"file '{file_path}'" for file_path in video_files])
    with open("output/video_list.txt", "w") as f:
        f.write(video_list)
    os.system(f"ffmpeg -y -loglevel error -f concat -safe 0 -i output/video_list.txt -c copy output/all_letters.mp4")

    # delete the video files
    for file_path in video_files:
        os.remove(f"output/{file_path}")



if __name__ == "__main__":
    main()

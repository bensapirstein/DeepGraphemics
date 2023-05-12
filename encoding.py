import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from fontTools.ttLib import TTFont, TTLibError
from tqdm import tqdm

class FontGraphemeAnalyzer:

    def __init__(self, threshold=0.5):
        self.scripts = pd.read_csv("data/scripts.csv", index_col=0)
        # check if font_encoding.csv exists
        if os.path.exists("data/font_encoding.csv"):
            self.font_encoding = pd.read_csv("data/font_encoding.csv", index_col=0)


        else:
            etymology_table = pd.read_csv("data/etymology_table.csv", index_col=0).T
            specific_encoding = pd.read_csv("data/specific_encodings_T.csv", index_col=0)

            self.encoding = pd.concat([etymology_table, specific_encoding], axis=0)
            self.font_encoding = pd.DataFrame(columns=["font_path", "script"] + self.encoding.columns.tolist())
            self.font_files = self.find_font_files()
            self.missing_glyphs = self.find_missing_glyphs()

            nans_percentage = self.font_encoding.isna().sum(axis=1) / self.font_encoding.shape[1]
            rows_to_remove = nans_percentage[nans_percentage > threshold].index.tolist()
            self.font_encoding = self.font_encoding.drop(rows_to_remove, axis=0)

            # save the font encoding to a csv file
            self.font_encoding.to_csv("data/font_encoding.csv")

        font_paths = self.font_encoding["font_path"].unique().tolist()
        self.load_fonts(font_paths)

    def find_font_files(self):
        """
        Find all font files in the data/fonts directory and return a dictionary
        """
        font_files = {}
        data_dir = "data/fonts"
        for script in self.scripts.index:
            script_dir = os.path.join(data_dir, script)
            font_files[script] = {}
            for encoding in os.listdir(script_dir):
                # check if the encoding is a folder
                if not os.path.isdir(os.path.join(script_dir, encoding)):
                    continue
                font_files[script][encoding] = []
                # look recursively for font files in fonts_dir using os.walk
                encoding_dir = os.path.join(script_dir, encoding)
                for root, dirs, files in os.walk(encoding_dir):
                    for file in files:
                        if file.endswith(".ttf") or file.endswith(".otf"):
                            font_path = os.path.join(root, file)
                            font_files[script][encoding].append(font_path)
                            font_manager.fontManager.addfont(font_path)

        # look for ttf files in the data_dir directory, they are multilingual fonts
        font_files["Multilingual"] = {}
        font_files["Multilingual"]["Unicode"] = []
        for file in os.listdir(data_dir):
            if file.endswith(".ttf") or file.endswith(".otf"):
                font_path = os.path.join(data_dir, file)
                font_files["Multilingual"]["Unicode"].append(font_path)
                font_manager.fontManager.addfont(font_path)

        return font_files

    def find_missing_glyphs(self, threshold=6): # TODO: Change threshold to percentage
        missing_glyphs = {}
        for script in tqdm(self.font_files.keys()):
            if script == "Multilingual":
                continue
            for encoding in self.font_files[script]:
                for font_file in self.font_files[script][encoding]:
                    font_name = os.path.splitext(os.path.basename(font_file))[0]
                    # add an entry to font_encoding for the current font, with the font path and the script
                    self.font_encoding.loc[font_name] = [font_file, script] + [None] * len(self.encoding.columns)
                    missing_glyphs[font_name] = []
                    if encoding == "Unicode" or encoding not in self.encoding.index:
                        self.check_glyphs(script, font_name, missing_glyphs)
                    else:
                        self.check_glyphs(encoding, font_name, missing_glyphs)

                    if len(missing_glyphs[font_name]) > threshold:
                        print(f"Missing glyphs in {font_file}:\n{', '.join(missing_glyphs[font_name])}")
        return missing_glyphs

    def load_fonts(self, font_paths):
        for font_file in font_paths:
            font_manager.fontManager.addfont(font_file)

    def is_letter_supported(self, letter, font_path):
        try:
            font = TTFont(font_path)
            glyph_name = font.getBestCmap().get(ord(letter))
            return glyph_name is not None
        except TTLibError:
            return False

    def check_glyphs(self, encoding, font_name, missing_glyphs):
        for letter in self.encoding.columns:
            letter_allographs = self.encoding.loc[encoding, letter]
            if pd.isna(letter_allographs):
                continue
            for grapheme_unicode in letter_allographs:
                if not self.is_letter_supported(grapheme_unicode, self.font_encoding.loc[font_name, "font_path"]):
                    missing_glyphs[font_name].append(grapheme_unicode)
                else:
                    if pd.isna(self.font_encoding.loc[font_name, letter]):
                        self.font_encoding.loc[font_name, letter] = grapheme_unicode
                    else:
                        self.font_encoding.loc[font_name, letter] += grapheme_unicode



    def plot_graphemes(self, letter, scripts, n_cols=10):
        # if script is a list
        script_fonts = self.font_encoding.loc[self.font_encoding["script"].isin(scripts)]

        N = script_fonts[letter].apply(lambda x: len(x) if not pd.isna(x) else 0).sum()
        n_rows = N // n_cols + 1
        fig = plt.figure(figsize=(12, 12))

        # turn off axis labels
        plt.axis('off')

        i = 0
        for font_name in script_fonts.index:
            font_file = script_fonts.loc[font_name, "font_path"]
            font = font_manager.FontProperties(fname=font_file)
            graphemes_unicodes = script_fonts.loc[font_name, letter]

            if pd.isna(graphemes_unicodes):
                continue

            for grapheme_unicode in graphemes_unicodes:
                if pd.isna(grapheme_unicode):
                    continue
                x = (i % n_cols) / n_cols
                y = 1 - ((i // n_cols) + 1) / n_rows

                plt.text(x, y, grapheme_unicode, fontproperties=font, fontsize=40, ha='center', va='center')
                i += 1

        plt.show()

def plot_font(font_file, script, graphemes, encoding):
    # get all the letters for the given script
    letters = graphemes.index.tolist()

    font_name = os.path.splitext(os.path.basename(font_file))[0]
    font = font_manager.FontProperties(fname=font_file)

    # create a grid of plots for all the letters
    n_cols = 10
    n_rows = len(letters) // n_cols + 1

    plt.axis('off')

    for i, letter in enumerate(letters):
        # get letter unicode from graphemes table
        if font_name in encoding.columns:
            letter_allographs = encoding.loc[letter, font_name]
        else:
            letter_allographs = graphemes.loc[letter, script]

        if pd.isna(letter_allographs):
           continue

        grapheme_unicode = letter_allographs[0]

        if not is_letter_supported(grapheme_unicode, font_file):
            x = grapheme_unicode.encode("ascii", "namereplace").decode("ascii")
            print(f"Glyph {grapheme_unicode} ({x}) missing from {font_name}.")
            continue


        x = (i % n_cols) / n_cols
        y = 1 - ((i // n_cols) + 1) / n_rows

        plt.text(x, y, grapheme_unicode, fontproperties=font, fontsize=40, ha='center', va='center')

    plt.suptitle(f"{script} - {font_file}")
    plt.show()


def main():
    fga = FontGraphemeAnalyzer()

    # plot_font("data/fonts/Paleo-Hebrew/Latin/PaleoBora.ttf", "Proto-Sinaitic", graphemes, encoding)
    #
    # for script in scripts.index:
    #     for font in font_files[script]:
    #         font_name = os.path.splitext(os.path.basename(font))[0]
    #         print(font_name)
    #
    #         # TODO: if font in encoding.columns perform encoding here and send the font specific unicode graphemes to plot_font
    #         plot_font(font, script, graphemes, encoding)

    scripts = ["Paleo-Hebrew", "Phoenician", "Proto-Sinaitic", "Samaritan", "Aramaic"]

    for letter in fga.font_encoding.columns[2:]:
        fga.plot_graphemes(letter, scripts, n_cols=15)

if __name__ == '__main__':
    main()

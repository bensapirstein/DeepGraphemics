import os
import pandas as pd
import skia
import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self, font_encoding_file, output_dir, fonts_dir, img_size=28, translate=True, 
                 rotation_dist=(0,5), sizes=range(20, 25), max_augmentations=10, repetitions=1, selected_scripts=None, selected_letters=None):
        """
        A class for generating a dataset of images from fonts for various scripts.
        """
        # Initialize configurable parameters
        self.output_dir = output_dir
        self.fonts_dir = fonts_dir
        self.img_size = img_size
        self.translate = translate
        self.rotation_dist = rotation_dist
        self.font_color = (255, 255, 255) # white
        self.back_color = (0, 0, 0) # black
        self.font_size_range = sizes
        self.max_augmentations = max_augmentations
        self.repetitions = repetitions

        # Load font encoding data
        self.font_encoding = pd.read_csv(font_encoding_file, index_col=0)

        # Select scripts and letters
        self.scripts = selected_scripts if selected_scripts else self.font_encoding.script.unique()
        self.letters = selected_letters if selected_letters else self.font_encoding.columns[2:]

        # Generating unique colors for each script
        script_colors = {}  # Dictionary to map each script to a color
        color_palette = plt.cm.tab10.colors
        for idx, script in enumerate(self.scripts):
            script_colors[script] = color_palette[idx % len(color_palette)]
        self.colors = [script_colors[script] for script in self.scripts]

        # Create a dictionary of fonts by script
        self.fnts = {}
        # Count the number of graphemes for each script and letter combination
        self.graphemes_count = pd.DataFrame(0, index=self.letters, columns=self.scripts)
        self.counts = {letter: 0 for letter in self.letters}

        # Load fonts and calculate grapheme and augmentation counts
        self.load_fonts()
        self.calculate_grapheme_counts()
        self.calculate_augmentation_counts()

    def load_fonts(self):
        """
        Load fonts from font files and organize them by script.
        """
        for font_name in self.font_encoding.index:
            font_path = self.font_encoding.loc[font_name, "font_path"]
            font_path = os.path.join(self.fonts_dir, font_path)
            script = self.font_encoding.loc[font_name, "script"]
            if script in self.scripts:
                if script not in self.fnts:
                    self.fnts[script] = {}
                font = skia.Font()
                font.setTypeface(skia.Typeface.MakeFromFile(font_path))
                self.fnts[script][font_name] = font

    def calculate_grapheme_counts(self):
        """
        Calculate the number of graphemes for each script and letter combination.
        """
        for font_name in self.font_encoding.index:
            for letter in self.letters:
                graphemes = self.font_encoding.loc[font_name, letter]
                script = self.font_encoding.loc[font_name, "script"]
                if pd.notna(graphemes) and script in self.scripts:
                    self.graphemes_count.loc[letter, script] += len(graphemes)

    def calculate_augmentation_counts(self):
        # create a matrix for augmentations for each letter and script
        G = self.graphemes_count
        augmentations = (1/G.div(G.max(axis=1), axis=0)).apply(np.ceil)
        
        # bound the number of augmentations by self.max_augmentations
        augmentations[augmentations > self.max_augmentations] = self.max_augmentations

        self.augmentations = augmentations.astype(int)

    def plot_graphemes_count(self):
        # sum the total number of graphemes for each script across all letters
        counts = np.array(list(self.graphemes_count.sum(axis=0).values))
        augments = np.array(list((self.graphemes_count * self.augmentations * self.repetitions).sum(axis=0).values))

        fig, ax = plt.subplots()
        ax.bar(self.scripts, counts, color=self.colors)
        ax.bar(self.scripts, augments, color=self.colors, alpha=0.5)
        # align the labels to right
        ax.set_xticklabels(self.scripts, rotation=45, ha='right')
        ax.set_ylabel('number of graphemes')
        ax.set_title('Number of graphemes for each script')
        plt.show()


    def generate_dataset(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            print(f"Output directory {self.output_dir} already exists. Exiting...")
            return

        surface = skia.Surface(self.img_size, self.img_size)
        paint = skia.Paint()
        with surface as canvas:
            for letter in self.letters:
                print(letter)
                for script in self.fnts:
                    path = os.path.join(self.output_dir, letter, script)
                    if not os.path.exists(path):
                        os.makedirs(path)

                    n_augmentations = self.augmentations.loc[letter, script] * self.repetitions

                    for font_name, font in self.fnts[script].items():
                        graphemes_unicodes = self.font_encoding.loc[font_name, letter]
                        if pd.isna(graphemes_unicodes):
                            continue
                        for i, character in enumerate(graphemes_unicodes):
                            if pd.isna(character):
                                continue
                            for j in range(n_augmentations):
                                self.random_transformations(canvas, letter, script, character, font, font_name, paint)
                            self.counts[letter] += n_augmentations
                print(f"Finished letter {letter}")
        print('Finished')
        print(f"total number of images generated: {sum(self.counts.values())}")
        print(f"Number of images generated for each letter: {self.counts}")

    def random_transformations(self, canvas, letter, script, character, font, font_name, paint):
        """
        Apply random transformations to the image.

        Parameters:
            canvas (skia.Surface): The surface to draw on.
            text_blob (skia.TextBlob): The text blob representing the glyph.
            bound (skia.Rect): The bounding box of the glyph.
            paint (skia.Paint): The paint object for drawing text.
            img_path (str): The path to save the generated images.
            img_name (str): The base name for the generated images.
            count (int): The number of augmentations to apply (default is 1).

        """
        canvas.clear(skia.Color(*self.back_color))

        paint.setColor(skia.Color(*self.font_color))
        # select a random font size
        size_value = np.random.choice(self.font_size_range)
        font.setSize(size_value)
        glyph = font.textToGlyphs(character)[0]
        bound = font.getBounds([glyph])[0]
        # center position
        position = (self.img_size - bound.width()) / 2 - bound.x(), (
                    self.img_size - bound.height()) / 2 - bound.y()

        builder = skia.TextBlobBuilder()
        builder.allocRunPos(font, [glyph], [position])
        text_blob = builder.make()

        # rotate the canvas by a random angle between -45 and 45 degrees, use normal distribution
        angle = np.random.normal(*self.rotation_dist)
        rotation = skia.Matrix()
        rotation.setRotate(angle, self.img_size / 2, self.img_size / 2)
        inverse_rotation = skia.Matrix()
        inverse_rotation.setRotate(-angle, self.img_size / 2, self.img_size / 2)

        # apply rotation to the bound
        rotated_bound = rotation.mapRect(bound)


        if self.translate:
            # translate the canvas by a random amount in both x and y directions
            # the maximum translation is based on the glyph width and height and the img_size
            pad = 1
            max_x = int((self.img_size - rotated_bound.width()) / 2 - pad)
            max_y = int((self.img_size - rotated_bound.height()) / 2 - pad)
            x = np.random.randint(-max_x, max_x) if max_x > 0 else 0
            y = np.random.randint(-max_y, max_y) if max_y > 0 else 0
        else:
            x, y = 0, 0

        canvas.concat(rotation)
        canvas.drawTextBlob(text_blob, x, y, paint)
        canvas.concat(inverse_rotation)

        # save the image
        surface = canvas.getSurface()
        image = surface.makeImageSnapshot()

        # augment the image
        img_name = f"{letter}_{script}_{font_name}_{character}_{size_value}_{angle}"

        img_path = os.path.join(self.output_dir, letter, script)

        image.save(os.path.join(img_path, f"{img_name}.png"))

def main():
    selected_scripts = ["Hieroglyph", "Proto-Sinaitic", "Phoenician", "Ancient North-Arabian",
                        "Ancient South-Arabian", "Ge'ez", "Paleo-Hebrew", "Samaritan", "Aramaic", "Syriac",
                        "Hebrew", "Nabataean", "Arabic"]
    dataset_generator = DatasetGenerator("../data/encoding/font_encoding.csv",
                                         f"../datasets/font_dataset",
                                         "../data/fonts",
                                         selected_scripts=selected_scripts)
    dataset_generator.generate_dataset()

if __name__ == '__main__':
    main()
import os
import pandas as pd
import skia
import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self, font_encoding_file, output_dir, fonts_dir, img_size=28, rotation_mean=0, rotation_std=5,
                 sizes=range(20, 25), max_augmentations=10, selected_scripts=None, selected_letters=None):
        """
        A class for generating a dataset of images from fonts for various scripts.
        """
        # Initialize configurable parameters
        self.output_dir = output_dir
        self.fonts_dir = fonts_dir
        self.img_size = img_size
        self.rotation_ = rotation_mean
        self.rotation_std = rotation_std
        self.font_color = (255, 255, 255) # white
        self.back_color = (0, 0, 0) # black
        self.font_size_range = sizes
        self.max_augmentations = max_augmentations

        # Load font encoding data
        self.font_encoding = pd.read_csv(font_encoding_file, index_col=0)

        # Select scripts and letters
        self.scripts = selected_scripts if selected_scripts else self.font_encoding.script.unique()
        self.letters = selected_letters if selected_letters else self.font_encoding.columns[2:]

        # Create a dictionary of fonts by script
        self.fnts = {}
        # Count the number of graphemes for each script
        self.graphemes_count = {letter: {script: 0 for script in self.scripts} for letter in self.letters}
        self.counts = {letter: 0 for letter in self.letters}

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
            if script in self.selected_scripts:
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
                    self.graphemes_count[letter][script] += len(graphemes)

    def calculate_augmentation_counts(self):
        """
        Calculate the number of augmentations required for each script and letter combination so that the number of
        augmentations times the number of graphemes for each script and letter combination is equal among all letters.
        We bound the number of augmentations by the maximum specified in self.max_augmentations.
        """
        self.augmentations = {}
        for letter in self.letters:
            counts = np.array(list(self.graphemes_count[letter].values()))
            augmentations = np.ceil(counts.max() / counts).astype(int)

            # Limit the number of augmentations to the specified maximum
            augmentations[augmentations > self.max_augmentations] = self.max_augmentations
            self.augmentations[letter] = {script: augmentation for script, augmentation in
                                          zip(self.graphemes_count[letter].keys(), augmentations)}

    def plot_graphemes(self, graphemes_count, counts, augmentations):
        # plot the number of graphemes for each script and the number of augmentations times the number of graphemes
        # for each script
        fig, ax = plt.subplots()
        ax.bar(self.augmentations.keys(), counts * augmentations, label='augmentations', alpha=0.5)
        ax.bar(graphemes_count.keys(), counts, label='graphemes', alpha=0.5)
        ax.set_xticklabels(graphemes_count.keys(), rotation=45)
        ax.set_ylabel('number of graphemes')
        ax.set_title('Number of graphemes for each script')
        ax.legend()
        plt.show()

    def generate_dataset(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        surface = skia.Surface(self.img_size, self.img_size)
        paint = skia.Paint()
        with surface as canvas:
            for letter in self.letters:
                print(letter)
                for script in self.fnts:
                    for font_name, font in self.fnts[script].items():
                        graphemes_unicodes = self.font_encoding.loc[font_name, letter]
                        if pd.isna(graphemes_unicodes):
                            continue
                        for i, character in enumerate(graphemes_unicodes):
                            if pd.isna(character):
                                continue
                            for size_value in self.font_size_range:
                                paint.setColor(skia.Color(*self.font_color))
                                font.setSize(size_value)

                                glyph = font.textToGlyphs(character)[0]
                                bound = font.getBounds([glyph])[0]
                                # center position
                                position = (self.img_size - bound.width()) / 2 - bound.x(), (
                                            self.img_size - bound.height()) / 2 - bound.y()

                                builder = skia.TextBlobBuilder()
                                builder.allocRunPos(font, [glyph], [position])
                                text_blob = builder.make()

                                # augment the image
                                image_name = f"{letter}_{size_value}_{script}_{font_name}_{i}"

                                image_path = os.path.join(self.output_dir, letter, script)

                                n_augmentations = self.augmentations[letter][script]
                                self.random_transformations(canvas, text_blob, bound, paint, image_path, image_name,
                                                            count=n_augmentations)
                                self.counts[letter] += n_augmentations
                print(f"Finished letter {letter}")
        print('Finished')
        print(f"Number of images generated for each letter: {self.counts}")

    def random_transformations(self, canvas, text_blob, bound, paint, img_path, img_name, count=1):
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
        for i in range(count):
            # apply random transformations
            canvas.clear(skia.Color(*self.back_color))
            # rotate the canvas by a random angle between -45 and 45 degrees, use normal distribution
            angle = np.random.normal(self.rotation_mean, self.rotation_std)
            rotation = skia.Matrix()
            rotation.setRotate(angle, self.img_size / 2, self.img_size / 2)
            inverse_rotation = skia.Matrix()
            inverse_rotation.setRotate(-angle, self.img_size / 2, self.img_size / 2)

            # apply rotation to the bound
            rotated_bound = rotation.mapRect(bound)

            # translate the canvas by a random amount in both x and y directions
            # the maximum translation is based on the glyph width and height and the img_size
            pad = 1
            max_x = int((self.img_size - rotated_bound.width()) / 2 - pad)
            max_y = int((self.img_size - rotated_bound.height()) / 2 - pad)
            x = np.random.randint(-max_x, max_x) if max_x > 0 else 0
            y = np.random.randint(-max_y, max_y) if max_y > 0 else 0

            canvas.concat(rotation)
            canvas.drawTextBlob(text_blob, x, y, paint)
            canvas.concat(inverse_rotation)

            # save the image
            surface = canvas.getSurface()
            image = surface.makeImageSnapshot()

            if not os.path.exists(img_path):
                os.makedirs(img_path)

            image.save(os.path.join(img_path, f"{img_name}_{i}.png"))

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
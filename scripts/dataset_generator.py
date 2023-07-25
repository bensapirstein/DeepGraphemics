import os
import pandas as pd
import skia
import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self):
        self.font_color_value = (255, 255, 255)  # white
        self.back_color_value = (0, 0, 0)  # black
        self.sizes = list(range(20, 25, 1))
        self.font_encoding = pd.read_csv("../data/encoding/font_encoding.csv", index_col=0)
        self.letters = self.font_encoding.columns[2:]
        selected_scripts = ["Hieroglyph", "Proto-Sinaitic", "Phoenician", "Ancient North-Arabian",
                            "Ancient South-Arabian", "Ge'ez", "Paleo-Hebrew", "Samaritan", "Aramaic", "Syriac",
                            "Hebrew", "Nabataean", "Arabic"]
        # variable to count the number of generated images for each letter
        self.counts = {letter: 0 for letter in self.letters}

        # create a dictionary of fonts by script
        self.fnts = {}
        # count the number of graphemes for each script
        graphemes_count = {letter: {script: 0 for script in selected_scripts} for letter in self.letters}
        for font_name in self.font_encoding.index:
            font_path = self.font_encoding.loc[font_name, "font_path"]
            script = self.font_encoding.loc[font_name, "script"]
            if script in selected_scripts:
                if script not in self.fnts:
                    self.fnts[script] = {}
                font = skia.Font()
                font.setTypeface(skia.Typeface.MakeFromFile(font_path))
                self.fnts[script][font_name] = font
                for letter in self.letters:
                    graphemes = self.font_encoding.loc[font_name, letter]
                    if not pd.isna(graphemes):
                        graphemes_count[letter][script] += len(graphemes)

        # augmentation parameters
        max_augmentations = 10
        self.rotation_mean = 0
        self.rotation_std = 45

        # calculate the number of augmentations for each script and letter
        self.augmentations = {}
        for letter in self.letters:
            counts = np.array(list(graphemes_count[letter].values()))
            augmentations = np.ceil(counts.max() / counts).astype(int)

            # limit the number of augmentations to 10
            augmentations[augmentations > max_augmentations] = max_augmentations
            self.augmentations[letter] = {script: augmentation for script, augmentation in
                                            zip(graphemes_count[letter].keys(), augmentations)}

        self.img_size = 28
        self.font_dir = f'../datasets/dataset_rotation_std_{self.rotation_std}/'

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
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)

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
                            for size_value in self.sizes:
                                paint.setColor(skia.Color(*self.font_color_value))
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

                                image_path = os.path.join(self.font_dir, letter, script)

                                n_augmentations = self.augmentations[letter][script]
                                self.random_transformations(canvas, text_blob, bound, paint, image_path, image_name,
                                                            count=n_augmentations)
                                self.counts[letter] += n_augmentations


            print(f"Finished letter {letter}")

        print('Finished')
        print(f"Number of images generated for each letter: {self.counts}")

    def random_transformations(self, canvas, text_blob, bound, paint, img_path, img_name, count=1):
        for i in range(count):
            # apply random transformations
            canvas.clear(skia.Color(*self.back_color_value))
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
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_dataset()

if __name__ == '__main__':
    main()
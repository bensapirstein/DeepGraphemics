import os
import numpy as np
import pandas as pd
import skia


class FontDatasetGenerator:
    def __init__(self):
        self.colors = {
            'red': (220, 20, 60), 'orange': (255, 165, 0), 'yellow': (255, 255, 0),
            'green': (0, 128, 0), 'cyan': (0, 255, 255), 'blue': (0, 0, 255),
            'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
            'silver': (192, 192, 192)
        }
        self.font_color_value = (255, 255, 255)  # white
        self.back_color_value = (0, 0, 0)  # black
        self.sizes = {
            'small': 10, 'medium': 13, 'large': 16, 'xlarge': 19, 'xxlarge': 22, 'xxxlarge': 25
        }
        self.font_encoding = pd.read_csv("data/font_encoding.csv", index_col=0)
        self.letters = self.font_encoding.columns[2:]
        self.img_size = 28
        self.max_w_len = 1  # Maximum word length
        self.font_dir = './dataset_skia'

    def generate_dataset(self):
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)

        for letter in self.letters:
            self.process_letter(letter)

        print('Finished')

    def process_letter(self, letter):
        print(letter)
        for size_name, size_value in self.sizes.items():
            print(size_name)
            for font_name in self.font_encoding.index:
                font_path = self.font_encoding.loc[font_name, "font_path"]
                script = self.font_encoding.loc[font_name, "script"]
                graphemes_unicodes = self.font_encoding.loc[font_name, letter]
                if pd.isna(graphemes_unicodes):
                    continue
                self.process_graphemes(graphemes_unicodes, font_name, font_path, letter, script, size_value)
        print('Finished processing letter:', letter)

    def process_graphemes(self, graphemes_unicodes, font_name, font_path, letter, script, size_value):
        for i, gu in enumerate(graphemes_unicodes):
            if pd.isna(gu):
                continue
            try:
                self.draw_grapheme(gu, font_path, size_value, letter, font_name, i, script)
            except Exception as e:
                print(f"Error: {gu}, {size_value}, {font_name}")
                print(e)

    def draw_grapheme(self, gu, font_path, size_value, letter, font_name, i, script):
        surface = skia.Surface(self.img_size, self.img_size)
        with surface as canvas:
            canvas.clear(skia.Color(*self.back_color_value))

            paint = skia.Paint()
            paint.setARGB(255, *self.font_color_value)

            font = skia.Font()
            font.setSize(size_value)
            font.setTypeface(skia.Typeface.MakeFromFile(font_path))

            text_blob = skia.TextBlob.MakeFromString(gu, font)

            # calculate x, y to align the text in the center, take under consideration the text bounds and current x y position
            x = (self.img_size - text_blob.bounds().width()) / 2 - text_blob.bounds().x()
            y = (self.img_size - text_blob.bounds().height()) / 2 - text_blob.bounds().y()

            # rotate the canvas by 45 degrees
            #canvas.rotate(np.random.randint(-45, 45), self.img_size / 2, self.img_size / 2)

            canvas.drawTextBlob(text_blob, x, y, paint)

            img_name = f"{letter}_{size_value}_{script}_{font_name}_{i}.png"
            img_path = os.path.join(self.font_dir, letter)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            image = surface.makeImageSnapshot()
            image.save(os.path.join(img_path, img_name))


    # a function to generate a single instance specified by the parameters
    def generate_instance(self, font_name, letter, size_value):
        font_path = self.font_encoding.loc[font_name, "font_path"]
        script = self.font_encoding.loc[font_name, "script"]
        graphemes_unicodes = self.font_encoding.loc[font_name, letter]
        if pd.isna(graphemes_unicodes):
            return
        self.process_graphemes(graphemes_unicodes, font_name, font_path, letter, script, size_value)

dataset_generator = FontDatasetGenerator()
#dataset_generator.generate_dataset()
dataset_generator.font_dir = './dataset_test'
dataset_generator.generate_instance('Lateef-Regular', 'ʿayn', 13)
dataset_generator.generate_instance('NotoNaskhArabic-VariableFont_wght', 'ʿayn', 13)
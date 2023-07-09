import os
import pandas as pd
import skia
import numpy as np

class FontDataset:
    def __init__(self):
        self.font_color_value = (255, 255, 255)  # white
        self.back_color_value = (0, 0, 0)  # black
        self.sizes = {
            'small': 10, 'medium': 13, 'large': 16, 'xlarge': 19, 'xxlarge': 22, 'xxxlarge': 25
        }
        self.font_encoding = pd.read_csv("data/encoding/font_encoding.csv", index_col=0)
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

                for i, gu in enumerate(graphemes_unicodes):
                    if pd.isna(gu):
                        continue


                    draw_grapheme(gu, font_path, size_value, letter, font_name, i, script)

        print('Finished processing letter:', letter)

# export draw_grapheme function outside the class
def draw_grapheme(character, font_path, size_value, letter, font_name, i, script):
    img_size = 28
    surface = skia.Surface(img_size, img_size)
    with surface as canvas:
        canvas.clear(skia.Color(0, 0, 0)) # black background

        paint = skia.Paint()
        paint.setARGB(255, 255, 255, 255) # white text

        font = skia.Font()
        font.setSize(size_value)
        font.setTypeface(skia.Typeface.MakeFromFile(font_path))

        text_blob = skia.TextBlob.MakeFromString(character, font)

        # calculate x, y to align the text in the center, take under consideration the text bounds and current x y position
        x = (img_size - text_blob.bounds().width()) / 2 - text_blob.bounds().x()
        y = (img_size - text_blob.bounds().height()) / 2 - text_blob.bounds().y()

        # rotate the canvas by 45 degrees
        canvas.rotate(np.random.randint(-45, 45), img_size / 2, img_size / 2)

        canvas.drawTextBlob(text_blob, x, y, paint)

        # draw a rectangle around the shifted text blob
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(1)
        paint.setARGB(255, 255, 0, 0)
        rect = skia.Rect.MakeXYWH(x + text_blob.bounds().x(), y + text_blob.bounds().y(), text_blob.bounds().width(), text_blob.bounds().height())
        canvas.drawRect(rect, paint)
        image = surface.makeImageSnapshot()

        img_name = f"{letter}_{size_value}_{script}_{font_name}_{i}.png"
        img_path = os.path.join('./dataset_skia', letter)
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        image.save(os.path.join(img_path, img_name))

    # a function to generate a single instance specified by the parameters
    def generate_instance(self, font_name, letter, size_value):
        font_path = self.font_encoding.loc[font_name, "font_path"]
        script = self.font_encoding.loc[font_name, "script"]
        graphemes_unicodes = self.font_encoding.loc[font_name, letter]
        if pd.isna(graphemes_unicodes):
            return
        self.process_graphemes(graphemes_unicodes, font_name, font_path, letter, script, size_value)

def main():
    dataset_generator = FontDataset()
    dataset_generator.generate_dataset()

if __name__ == '__main__':
    main()
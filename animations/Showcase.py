from coldtype import *
from coldtype.fx.skia import phototype
from easing_functions.easing import EasingBase

import coldtype.timing.easing as easing
import pandas as pd

# load scripts from csv file
scripts = pd.read_csv("data/scripts.csv", index_col=0)
graphemes = pd.read_csv("data/minimal_etymology_table.csv", index_col=0)

selected_scripts = ["Hieroglyph", "Proto-Sinaitic", "Phoenician", "Ancient North-Arabian", "Ancient South-Arabian", "Ge'ez", "Paleo-Hebrew", "Samaritan", "Aramaic", "Syriac", "Hebrew", "Nabataean", "Arabic"]
selected_scripts = selected_scripts[2:]
M = len(selected_scripts)
fnts = []
for script in selected_scripts:
    fnts.append(Font.Cacheable("data/semiticRegular/" + scripts.loc[script, "font path"]))

titles = graphemes.index

fnt_lt = "assets/ColdtypeObviously-VF.ttf"

# Define the string of Phoenician letters
script = "Phoenician"
N = len(titles) 
fps = 60 # frames per second
secs = 2
fpl = secs * fps # frames per letter

class MyExponentialEaseInOut(EasingBase):
    def func(self, t, p=0.3):
        if t == 0 or t == 1:
            return t

        if t < 0.5:
            return 0.5 * math.pow(t * 2, p)
        return 1 - 0.5 * math.pow(2 - t * 2, p)
    
easing.eases["meeio"] = MyExponentialEaseInOut

@animation(timeline=(fpl*N, fps), bg=0)
def showcase_letters(f):
    # Calculate the index of the current frame
    i = f.i // fpl
    title = titles[i]

    j = min(int(f.e("qeio", loops=N/2, cyclic=False) * M), M - 1)
    script = selected_scripts[j]


    return (StSt(graphemes.loc[title, script], fnts[j], 800)
            .align(f.a.r)
            # translate so that slides from the right
            .translate(f.e("meeio", loops=N/2, rng=[1080, -1080], cyclic=False), -80)
            .f(1)
            .ch(phototype(f.a.r, blur=3, cut=90, cutw=30, fill=bw(1))),
            StSt(titles[i],fnt_lt,80)
            .align(f.a.r)
            .translate(0, 450)
            .f(1))
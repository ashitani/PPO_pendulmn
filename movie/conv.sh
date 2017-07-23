#!/bin/sh
ffmpeg -i %d.png -vf palettegen palette.png
ffmpeg -f image2 -r 30 -i %d.png -i palette.png -filter_complex paletteuse out.gif

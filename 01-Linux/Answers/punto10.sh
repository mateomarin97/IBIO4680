#!/bin/bash

for f in ./*.jpg
do
  mogrify -crop 256x256+0+0 $f
done

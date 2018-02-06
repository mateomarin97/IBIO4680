#!/bin/bash
let cont=0;
for f in ./../../imgs/BSR/BSDS500/data/images/test/*.jpg
do
  r=$(identify -format '%[fx:(h>w)]' "$f")
  if [[ r -eq 1 ]]
  then
      let cont=cont+1
  fi
done
echo Las imagenes lanscape son: $cont

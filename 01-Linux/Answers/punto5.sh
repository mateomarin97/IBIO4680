#!/bin/bash

for i in *.jpg
do
  let ck=`cksum $i|cut -d " " -f 1`
  for j in *.jpg
  do
    let ck2=`cksum $j|cut -d " " -f 1`
    if [ "$ck" == "$ck2" ] && [ "$i" != "$j" ] ; then
      echo Duplicados: $i $j
    fi
  done
done

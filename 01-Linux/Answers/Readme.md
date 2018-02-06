Respuestas al laboratorio 1:


1. imprime a consola todas las lineas que contienen el parametro especificado

2. Le especifica con que programa (bash) ejecutarlo

3. `cat /etc/passwd | wc -l`

4. `cut -f 1,7 -d ":" /etc/passwd | sort -t$":" -k 2`

5.
```bash
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

```
7.

`ls --block-size=KB BSR_bsds500.tgz` y `ls --block-size=KB imgs/BSR/BSDS500/data/images -Rli | grep ".jpg" |wc -ls`

8.


`for f in *; do file "$f"; done | grep "JPEG" | cut -d " " -f 1,2,18| tr "," " " | tr ":" " "`

9.
```bash
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
  ```

La funcion: `identify -format '%[fx:(h>w)]' "$f"` fue obtenida de: <https://unix.stackexchange.com/questions/294341/shell-script-to-separate-and-move-landscape-and-portrait-images>

10.
```bash
#!/bin/bash

for f in ./*.jpg
do
  mogrify -crop 256x256+0+0 $f
done
```

Comando mogrify obtenido de:
<https://codeyarns.com/2014/11/15/how-to-crop-image-using-imagemagick/>


Hecho en colaboración con Mauricio Neira codigo: 201424001

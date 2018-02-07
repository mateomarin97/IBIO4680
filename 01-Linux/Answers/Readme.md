Respuestas al laboratorio 1:


1. imprime a consola todas las lineas que contienen el parametro especificado

2. Le especifica con que programa (bash) ejecutarlo

3. `cat /etc/passwd | wc -l`

cat se encarga de imprimir en consola lo que sea que tenga el archivo que se le indica, en este caso se le indica imprimir el contenido del archivo passwd el cual contiene la información de los usuarios. Luego wc sirve para contar, el -l le especifica que cuente líneas y como cada línea es un usuario el resultado es el número de usuarios. Cuando lo corrimos en nuestra maquina dijo que habian 42 usuarios.

4. `cut -f 1,7 -d ":" /etc/passwd | sort -t$":" -k 2`

El comando cut sirve para imprimir en consola la información del archivo especificado pero sólo imprime las columnas de ese archivo que se espesifican, para eso es el -f, en este caso queremos que imprima la columna 1 (el nombre de usuario) y la 7 que da el shell, el -d especifica el delimitador de las columnas (en este caso las columnas están separadas por :), luego el sort toma estos datos y los organiza en orden alfabetico de la columna especificada, en este caso es la 2 que da las Shell, hay que volver a decirle el delimitador y para eso es el -t.

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

Bueno este script se debe correr dentro de la carpeta con las imagenes, el primer for agarra una por una cada una de las imagenes, se guarda en la variable ck el checksum del archivo, por eso se usa el cut para sólo agarrar el primer valor que imprime cksum, pues el segundo es el número de bytes del archivo. El segundo for agarra de nuevo cada una de las imagenes para poder comparar cada una de ellas con la imagen del primer for, se guarda en ck2 el checksum de la segunda imagen, por último el if valida que los dos archivos tengan el mismo checksum(tengan la misma información) pero que tengan nombres diferentes (para no comparar una imagen con sigo misma). Si se cumple el if entonces se imprime en la terminal los nombres de las dos imagenes duplicadas. 

7.

`ls -l --block-size=KB BSR_bsds500.tgz` y `ls --block-size=KB imgs/BSR/BSDS500/data/images -Rli | grep ".jpg" |wc -l`


ls -l muestra toda la información del archivo especificado, incluyendo su tamaño, el --block-size=KB se coloca para que exprese el tamaño en KB. Dice que el tamaño de BSR_bsds500.tgz es 70764 kB.

Por otro lado basicamente estamos imprimiendo todos los archivos que están en la carpeta imgs/BSR/BSDS500/data/images con toda su información, ya que esta carpeta tiene subcarpetas el comando debe ser recursivo y se coloca el -R (el -i es para poner un indice a cada imagen), luego usamos grep para sólo dejar los que tenga un .jpg en el nombre y finalmente wc -l para contar las líneas. Por cierto dice que hay 997 imagenes.

8.


`for f in *; do file "$f"; done | grep "JPEG" | cut -d " " -f 1,2,18| tr "," " " | tr ":" " "`

El for agarra cada archivo que haya en la carpeta y le ejecuta el comando file, que imprime en la terminal el nombre del archivo, el tipo de archivo (JPEG image data para las imagenes) y el número de pixeles o resolución (también imprime otras cosas pero no importan ahora). Se manda esto al grep para que sólo agarre las imagenes, al agarrar sólo lineas con la palabra JPEG en ellas, luego se manda al cut para agarrar sólo la primer columna(nombre del archivo), la segunda (tipo archivo) y la 18 (la resolución), los dos tr del final son para cambiar primero las comas por espacios vacios y luego los : por espacios vacios, son sólo cuestión de estetica.

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

Este script está hecho para mirar solo la carpeta test, pero se puede acomodar facilmente para mirar cualquier carpeta. Por ejemplo se puede usted parar en la carpeta deseada y sólo usar for f in *.jpg

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

La verdad ya no hay mucho que explicar sólo agarra todas las imagenes y les aplica el comando mogrify -crop axb+c+d que las corta a un tamaño de axb pixeles empezando en el pixel c,d.


Hecho en colaboración con Mauricio Neira codigo: 201424001

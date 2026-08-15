# Cómo empezar a trabajar
1. Abre una terminal en tu equipo o aquí en Visual
2. En la terminal revisa bien que estés dentro de la carpeta (que diga git)
3. Asegúrate de cambiarte a la rama de "develop"

```bash
git checkout develop
```

4. Tráete los últimos cambios antes de comenzar a trabajar

```bash
git pull
```


# Cómo subir tus cambios
1. Una vez que hayas terminado con tus cambios recuerda usar los siguientes comandos para subirlos a develop y que los demás puedan tener las cosas actualizadas.

```bash
git add .
git commit -m "(AQUÍ VAN LOS CAMBIOS QUE HICISTE — NO OLVIDES QUITAR LOS PARÉNTESIS)
git push
```
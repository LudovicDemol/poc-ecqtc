### Comment remplir la base de donnée vectorielle ?

- Placer les fichiers de données .txt dans le dossier data/docs

- Executer la ligne de commande : `curl -X POST http://localhost:8000/ingest $(for f in /data/docs/*.txt; do echo "-F files=@$f"; done) -F "namespace=default"`
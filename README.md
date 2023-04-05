# Pretrained-captioning
Service d'étiquetage d'image préentrainé.  


## Description
Le conteneur utilise des modèles préentraînés de reconnaissance d’image pour décrire, avec des mots et des phrases, le contenu de l’image. Deux modèles différents sont utilisés pour générer les phrases et les mots.
 
Le premier est un modèle de « Image Captioning », qui prend en entrée une image et retourne une phrase qui décrit son contenu. Afin d’utiliser ce modèle, nous utilisons la librairie LAVIS. Cette librairie permet d’importer, avec la fonction load_model_and_preprocess(), le nom du modèle de base à utiliser ainsi que les paramètres « fine-tuned » du modèle sur un ensemble de données spécifique. Pour une tâche de « Image Captioning », nous avons importé le modèle BLIP et, pour le type de modèle, nous avons utilisé la version « fine-tuned » de BLIP sur l’ensemble de données MSCOCO. Celui-ci est un ensemble de données de plus de 200 000 images avec 5 descriptions d’images par image.
 
Le modèle BLIP offre deux modes de fonctionnement différents. Le premier est une « Image captioning » par Beam Search. Ceci est une prédiction déterministe, donc qui génère toujours la même description pour une même image. Cette description est aussi généralement la plus précise. L’autre type de génération est la génération par Nucleus Sampling. Celle-ci est une prédiction probabiliste, donc qui génère toujours une description différente. Celle-ci peut être plus expressive que la génération de Beam Search, mais elle a plus de chance de commettre des erreurs.
 
Dans notre approche, nous utilisons les deux modes de fonctionnement. En premier, nous générons une phrase descriptive de l’image par beam search, que nous renvoyons comme réponse. Ensuite, nous générons une multitude de phrases (5 par défaut) par nucleus sampling. Ces phrases ont un vocabulaire plus varié que le beam search, et permettent donc de générer un vocabulaire de mots pour le prochain modèle.
 
Le deuxième modèle que nous utilisons est CLIP, un modèle fait par OpenAI qui permet de faire de la classification « Zero-Shot » avec des mots et une image. Ainsi, avec ce modèle, nous pouvons présenter une image avec une multitude de mots puis le modèle assignera à chaque mot un pourcentage qui indique la portion de l’image occupée par ce mot. Les mots que nous utilisons viennent du vocabulaire généré par BLIP avec Nucleus Sampling et par Beam Search.
 
Finalement, nous retournons dans un objet JSON tous les mots avec leur pourcentage, la description générée par Beam Search et les descriptions générées par Nucleus Sampling.
 
## Amélioration possible
Avec la librairie LAVIS, il est possible d’importer bien d’autres modèles que BLIP. Il est possible de voir, avec le code suivant, tous les modèles supportés par LAVIS. Les architectures qui nous intéressent pour du « Image captioning » sont dans le tableau ci-dessous. Nous avons décidé d’utiliser blip_caption avec base_coco, car ce modèle offrait un bon compromis entre la rapidité et la précision des captions. Par contre, des tests avec BLIP2 nous ont montré que ce modèle donne des descriptions encore plus précises et plus descriptives de l’image. Le modèle BLIP2 peut être essayé ici: https://huggingface.co/spaces/Salesforce/BLIP2 . On peut choisir beam search pour toujours avoir une même phrase ou nucleus sampling pour varier les phrases.
 
 
## installation et déploiement
Pour cloner le répertoire et apporter des modifications à l’image sur dockerhub:
-   	Installer git: sudo apt-get install git
-   	git clone https://github.com/WilliamYn/pretrained-captioning
-   	Apporter des modifications au code et push
-   	Une pipeline permettra à l’image sur dockerhub d’être reconstruite
 
Pour exécuter le conteneur docker localement (sur une machine Linux):
Installer docker:
sudo wget -O get-docker.sh https://get.docker.com/
sudo sh get-docker.sh
Télécharger et rouler le conteneur docker:
sudo docker pull walle123/flask_captioning:latest
sudo docker run -p 3000:80 -e caption_model_name=”blip_caption” -e caption_model_type_name=”base_coco” -e total_captions_number=”5”
 
Variables d’environnement du conteneur Docker:
-   	caption_model_name: le nom de l’architecture à utiliser. Par défaut, celui-ci est défini à “blip_caption”.
-   	caption_model_type_name: le nom du type d’architecture à utiliser. Par défaut, celui-ci et défini à “base_coco”.
-   	total_captions_number: le nombre de phrases à générer
 
## Architecture
Type
blip_caption
base_coco, large_coco
blip2_opt
pretrain_opt2.7b, pretrain_opt6.7b, caption_coco_opt2.7b, caption_coco_opt6.7b
blip2
pretrain, pretrain_vitL, coco

### Notes
Un autre modèle que nous avons essayé est vit-gpt2-image-captioning. Nous n’avons pas gardé ce modèle, car il donnait souvent des descriptions imprécises des images. Donc nous avons décidé de ne pas l’utiliser.

## Utilisation 
Il y a deux routes rendues disponibles par Flask.
GET /hello-world    -Pour vérifier l'état du service
POST /              -Pour évaluer les images et générer les étiquettes et les phrases descriptives

Pour la requete POST, voici les JSON en input et output
#### Input JSON
```
{
    "image": "image convertit en base64"
}
```
#### Output JSON
```
{
    "tags": [
        ["tag1", 0.38],
        ["tag2", 0.12],
        ...,
        ["tagX", 0.01]
    ],
    "captions": ["the best caption"]
    "english_captions": [
        "The sentence the tags were generated from.",
        ...,
        "Another sentence the tags were generated from."
    ]
}
```

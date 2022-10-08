#DVC - DL - TF - AIOPS demo
download data --> [source](https://drive.google.com/drive/folders/1BL-HD_nr_38JgfHKTlofxodBlKGeA_as?usp=sharing)
##open anacodna propmpt
##commands
``` bash

cd /d d:\
cd D:\DVC_DL_Tensorflow_demo
conda create ./env --prefix=3.7 -y
conda activate ./env
code
conda env list

pip install -r requirements.txt

git init

dvc init

touch README.md .gitignore setup.py dvc.yaml params.yaml

mkdir -p config src/utils 

touch config/congig.yaml config/secrets.yaml

touch src/__init__.py src/utils/__init__.py param.yaml dvc.yaml config/config.yaml src/stage_01_load_save.py src/utils/all_utils.py setup.py .gitignore

install src

pip install -e .

pip freeze > requirements.txt


```


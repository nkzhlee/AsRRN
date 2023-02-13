


conda list -e > requirements.txt

can be used to create a conda virtual environment with

conda create --name <env> --file requirements.txt
conda create --name py3.6_ASAG --file requirements.txt

If you want a file which you can use to create a pip virtual environment (i.e. a requirements.txt in the right format) you can install pip within the conda environment, then use pip to create requirements.txt.

conda activate <env>
conda install pip
pip freeze > requirements.txt
Then use the resulting requirements.txt to create a pip virtual environment:

conda create -n py3.6_ASAG python=3.6

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
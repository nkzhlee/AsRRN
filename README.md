# ASGA


Step 1: Create a New Conda Environment
First, you create a new Conda environment. You can specify the version of Python you want in the environment if your requirements.txt file is dependent on a specific Python version. If not specified, the default Python version that comes with your Anaconda installation will be used.

conda create -n mynewenv python=3.6

Step 2: Activate the New Environment
Before installing the packages, you need to activate the newly created environment.

conda activate mynewenv

Step 3: Install Packages Using pip
With the environment activated, you can now install the required packages listed in your requirements.txt file using pip.

pip install -r requirements.txt

Ensure that requirements.txt is the path to your requirements file. If it's in the current directory, the command as shown will work. If it's in another directory, you'll need to specify the full path to the file.

Step 4 Change the parameters in constant.py

You might need to change the wangdb user information if you want to use it, otherwise you need to comment all the code in main.py to disable wangdb.

Step 5 To run the code

python -u main.py --ckp_name ckp --device cuda:0 
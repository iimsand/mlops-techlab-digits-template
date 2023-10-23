#!/bin/bash

echo "export PATH=`python -m site --user-site`/bin:\$PATH" >> ~/.bashrc
echo "export PYTHONPATH=$PWD" >> ~/.bashrc

pip install --upgrade pip
pip install -r requirements.txt

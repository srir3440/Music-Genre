#!/bin/bash

sudo apt-get update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo apt install docker-compose
cd ./upload
sudo docker-compose build
sudo docker-compose up
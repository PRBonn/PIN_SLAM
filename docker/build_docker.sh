#!/bin/bash

echo "Build docker"

sudo docker build -f cu117.Dockerfile -t pinslam:localbuild .

echo "docker successfully build!"

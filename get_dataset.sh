#!/bin/sh

which unzip > /dev/null && which wget > /dev/null || { echo "Please install wget and unzip" && exit 1; }
wget "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "dataset.zip"
unzip dataset.zip
rm dataset.zip


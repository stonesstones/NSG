#!/bin/bash

DATADIR="/groups/gcd50654/tier4/neural-scene-graphs/example_weights"
TEMPFILE="/groups/gcd50654/tier4/neural-scene-graphs/tmp.zip"
wget -O $TEMPFILE "https://drive.google.com/uc?export=download&id=1o28o6gOGHrjQ3LA5Kazj6zdzXEVboS8g" && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE

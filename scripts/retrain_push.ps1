#!/bin/sh

python -m scripts.retrain_models

git add .
git commit -m 'Retrained models'
git push origin test

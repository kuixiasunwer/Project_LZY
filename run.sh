#!/bin/bash

if [ "$SKIP_TRAIN" = "TRUE" ]; then
    echo "Skipping training as SKIP_TRAIN is set to TRUE"
    echo "If you want to recurrence, make sure run docker with SKIP_TRAIN set to FALSE"

    echo "Try Create Target Dir"
    mkdir -p /saisresult
    echo "Start Copy"
    cp -p /app/final_output.zip /saisresult/output.zip
    echo "Pre Output Finish 2"
else
    echo "Program start!!!"
    echo "No Use History Data"

    pip3 install -r requirements.txt  --no-index --find-links=/tmp/packages
    pip3 uninstall -y lightgbm
    pip3 install /app/data/packet/lightgbm-4.6.0-py3-none-linux_x86_64.whl

    python code/main.py
fi
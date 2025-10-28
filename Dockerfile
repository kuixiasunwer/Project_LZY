FROM crpi-hya7su85ipw2xkib.cn-shenzhen.personal.cr.aliyuncs.com/ouc_china_lingxi/sais_ouc_iteration:1.1.Base
# This Docker Include the Temp of Pip

#FROM python:3.12-slim
LABEL authors="lingxi"


COPY /code/main.py /app/code/main.py

COPY /configs /app/configs

#COPY /data/log /app/data/log
#COPY /data/processed/np_data/ /app/data/processed/np_data/
COPY /data/packet/ /app/data/packet/
#COPY /data/processed/xr_data/extend_feature.nc /app/data/processed/xr_data/extend_feature.nc

COPY /draw /app/draw
COPY /models /app/models
COPY /output /app/output
COPY /pre /app/pre
COPY /train /app/train
COPY /util /app/util

COPY pyproject.toml /app
COPY requirements.txt /app
COPY run.sh /app

#COPY data/Input-train/ /saisdata/train/POWER_TRAIN_ENV
#COPY data/Input-train/fact_data /saisdata/train/power_train
#COPY data/Input-test/ /saisdata/test/POWER_TEST_ENV

WORKDIR /app

# IMPORTANT
ENV SAIS=TRUE

# MAKE SURE SCORE NO CHANGE (Cancel)
#COPY output_final.zip /app/final_output.zip
#ENV SKIP_TRAIN=TRUE

CMD ["bash", "run.sh"]
#CMD ["bash"]
#!/bin/bash

docker login --username=clodlingxi crpi-c4fefeszbgqmv9cc.cn-shanghai.personal.cr.aliyuncs.com --password=ouc_password_asd

echo "确认Dockerfile删除saisdata相关处理..."
#
docker build -t  crpi-c4fefeszbgqmv9cc.cn-shanghai.personal.cr.aliyuncs.com/ouc_docker/docker:1.0 .
docker push  crpi-c4fefeszbgqmv9cc.cn-shanghai.personal.cr.aliyuncs.com/ouc_docker/docker:1.0


#docker build -t  crpi-hya7su85ipw2xkib.cn-shenzhen.personal.cr.aliyuncs.com/ouc_china_lingxi/sais_ouc_iteration:0.30 .
#docker push  crpi-hya7su85ipw2xkib.cn-shenzhen.personal.cr.aliyuncs.com/ouc_china_lingxi/sais_ouc_iteration:0.30
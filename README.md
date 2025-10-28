# Sais

## Project Tree
- `configs`：配置文件目录
  - `base_config.py`
- `data`：数据目录
  - `Input-test`：测试输入数据（与初赛提交格式相同）
  - `Input-train`：训练输入数据（与初赛提交格式相同）
  - `log`：日志文件
  - `processed`：处理后的数据（缓存）
- `draw`：绘图脚本
  - `functions.py`
- `models`：模型目录
  - `base`：基础模型目录
  - `FixModel`:基础风电模型
  - `LSTM5_24.py`：LSTM 模型文件 （唯一使用模型）
- `output`：输出目录
- `pre`：预处理目录
  - `cleaner.py`：死值空值处理
  - `feather_engineering.py`：特征工程
  - `extend_dataset.ipynb` 复用初赛数据（未用）
  - `fix_label.ipynb` 手动处理死值（未用）
- `train`：训练目录
  - `functions.py`：训练函数
  - `tools.py`：训练工具
- `util`：工具目录
  - `data_loader.py`：数据加载器
  - `paths.py`：路径管理
- `code` 代码入口
  - `main.py`：主脚本
- `environment.yml`：**Conda环境配置文件（推荐使用）**    
- `pyproject.toml`：项目配置文件
- `README.md`：项目说明文档

## Environment1（刘梓阳）
- Python: 3.12
- Pytorch 2.7.0 
- Cuda 12.4
  
## Environment2（梅宇轩）
- Python: 3.11.13
- Pytorch 2.5.1
- Cuda 12.9
  
### Method 1 (梅)
```shell
# 确保工作目录为根目录
conda env create -f environment.yml
```
### Method 2 (刘Recommended)
```shell
python -m pip install uv
uv sync
```

### Method 3 (刘For Production)
- Make Sure Set `Env SAIS=TRUE`
```shell
pip install -r requirements.txt
```

## Run

```shell
# 确保工作目录为根目录
python code/main.py
```

## Reminder
程序在首次运行时，会启动 `swanlab` 的交互式设置向导，在终端出现以下选项时：
> ```
> swanlab: (1) Create a SwanLab account.
> swanlab: (2) Use an existing SwanLab account.
> swanlab: (3) Don't visualize my results.
> swanlab: Enter your choice:
> ```
-**输入'3'并回车**，以离线模式继续运行，无需注册或登录


## Update Log

`0.0 -> 0.1`：Online train \
`0.1 -> 0.2`：Train epoch Update \
`0.2 -> 0.3`：Rnn input size \
`0.3 -> 0.4`：Wind sped feature \
`0.4 -> 0.8`：Res Conv \
`0.8 -> 0.10`：Special Conv Decoder 2 resolution No shared \
`0.10 -> 0.11` Wind: Resolution 2, No Shared; Light: Resolution 4, Shared \
`0.11 -> 0.25` Make Conv Larger And Ensemble with 0.4 (Test Score Same to 0.11) \
`0.25 -> 1.0` Ensemble 0.25 And 0.11 (Draw In ./draw/final_output.png)


## Other Try
- [`ResNetSmallConvLSTMModel` And `ResNetConvLSTMModel` And `Sklearn`] Ensemble (Need More Try)

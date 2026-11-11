# Legacy Implementation Archive

本目录归档了原项目的基于 **Python 2.7 + TensorFlow 1.5 + Keras 2.2.4** 的历史代码，仅供历史对照与学术追溯参考。

## 归档文件说明
- `complete.py`: 原作者的 GPU 版本单文件脚本（强依赖 `CuDNNGRU`，与现代框架不兼容）。
- `complete_cpu.py`: 原作者的 CPU 临时版本（存在硬编码的用户本地绝对路径 `/home/masanoyu/...`）。
- `dgcnn.py`: 原 DGCNN 模块定义草稿。
- `load_data.py`: 原旧数据提取草稿脚本。
- `trainsfor.py`: 原旧数据预处理草稿（原文件名存在拼写错误 `trainsfor`）。
- `word2vec.py`: 原旧词向量加载与切词草稿。

> **提示**：现代生产及学术研究请使用项目根目录下的全新重构工程（基于 **PyTorch 2.x + 模块化工程架构**）。

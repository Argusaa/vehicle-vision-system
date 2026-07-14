# Vehicle Vision System

车载视觉感知与人机交互系统的主项目位于：

```text
database/vehicle-vision-system/
```

该项目包含四个业务模块：

- 车辆车牌识别
- 交警手势识别
- 车主手势控车
- 日志监控与告警智能体

## 克隆与启动

环境要求：Windows 10/11、64 位 Python 3.11、Git 与 Git LFS。推荐直接运行
一键启动脚本，它会创建 `.venv`、安装依赖、拉取模型、初始化 SQLite 与本机
HTTPS 证书。

```powershell
git lfs install
git clone https://github.com/Argusaa/vehicle-vision-system.git
cd vehicle-vision-system
git lfs pull
cd database/vehicle-vision-system
start.bat
```

首次安全初始化会在终端显示随机生成的 `admin` 密码，请立即保存。程序默认仅
监听 `127.0.0.1:8001`，数据库使用无需额外安装的 SQLite。

交警手势识别使用本项目训练的 `lstm_yolo11s.pt`，该权重由 Git LFS
分发。若模型校验提示文件仍为指针或不完整，请在仓库根目录运行
`git lfs pull`。

每台电脑首次运行会自动完成本机密钥、管理员和 HTTPS 证书初始化。
默认访问地址为 <https://localhost:8001>，API 文档位于
<https://localhost:8001/api/docs>。

详细功能、配置和目录说明请参阅
[`database/vehicle-vision-system/README.md`](database/vehicle-vision-system/README.md)。

仓库中的 `database/CCPD-master`、`database/ctpgr-pytorch-master` 和
`database/hagrid-master` 是各视觉模块使用或参考的数据集与模型项目。

## 许可证

本项目原创代码及自训练的 `lstm_yolo11s.pt` 使用 [MIT License](LICENSE)。
仓库内第三方目录和资产继续遵循各自许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

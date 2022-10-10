# 猫雷AI的Flask后端接口

基于[猫雷AI](https://github.com/IceKyrin/sovits_f0_infer)模型，封装了Flask接口，可以配合VST插件实现C/S架构的声音信号处理

## 使用方法

构建docker镜像

```bash
bash -xe build_docker.sh
```

通过docker-compose进行服务启动

```bash
docker-compose up -d
```

服务启动在6842端口

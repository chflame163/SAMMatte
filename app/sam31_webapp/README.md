# SAM 3.1 Web App

本地 Web App，交互方式参考 `sam2/demo`，后端推理改为 `sam3.1`。

## 启动

在 `E:\work\video_segment\SAMMatte` 下运行：

```powershell
.\run_SAMMatte.bat
```

默认会启动在 `http://127.0.0.1:8765`。

运行时缓存会写入 `cache\`，并在每次程序启动时自动清空上一次缓存。

模型源码和权重位于 `models\sam3\`、`models\sam3.1\` 与
`models\vitmatte-base-composition-1k\`。

## 当前功能

- 拖拽或选择本地视频上传
- 在任意帧用 `points`、`bbox` 或 `text prompt` 生成当前帧 mask
- 点选模式支持关键帧开关，可在不同帧确认不同多点组合，并可跳转上/下关键帧或删除当前关键帧；右键已有点可转换正负或删除；关闭时只保留最后一次点组选帧
- 关键帧模式下，右键当前帧已有点可通过上下文菜单编辑当前帧点组；已确认关键帧编辑后需再次预览确认
- 从该帧向前后双向传播
- 自动生成 overlay preview 和黑白 mask preview
- 在 mask 预览窗口右键，按 H.264 + 自定义码率导出

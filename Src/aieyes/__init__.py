"""AI_Eyes 视觉语言交互子系统（interaction.v0.1）。

代码组织与职责边界见 ``Src/aieyes/interfaces.py``；各子包分别负责：

* ``audio``   采集、播放、AEC、VAD、WAV 读写
* ``speech``  唤醒（KWS）、流式 ASR、TTS、声纹
* ``vision``  相机取图、OCR、视觉语言模型客户端、图像质量与几何
* ``contract`` interaction.v0.1 消息编码/校验、TCP 服务端、测试夹具
* ``interaction`` 会话状态机、计时、任务与证据存储（Lead 负责）
* ``llm``     llama.cpp server 的文本请求封装（Lead 负责）
* ``replay``   离线文件回放入口（阶段 0 的“不打开声卡或相机”路径）
"""

__version__ = "0.0.1"

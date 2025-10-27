# MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation

如图所示，MultiAgentESC框架由三个阶段构成。第一阶段是对话分析，由多个角色各异的智能体从对话上下文中提取用户的心理状态。第二阶段是策略商议，MultiAgentESC会从数据集中检索相似案例，并将其融入商议流程，以缓解大语言模型的偏好偏差，并在此后保留多个有效策略。第三阶段是回复生成，系统利用这些策略生成多样化的回复，最终通过多智能体间的辩论与协作，筛选出最佳回复。

![image](https://github.com/MindIntLab-HFUT/MultiAgentESC/blob/main/image/model4.png)

## Quick Start

#### 1. 克隆仓库
```bash
git clone https://github.com/MindIntLab-HFUT/MultiAgentESC.git
```

#### 2. 进入目录
```bash
cd MultiAgentESC
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 替换相关路径

替换main.py文件中parse_args()的相关路径

#### 5. 执行
```bash
python main.py
```

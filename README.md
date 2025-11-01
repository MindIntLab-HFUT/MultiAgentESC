# MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation

As demonstrated in the Figure, MultiAgentESC consists of three stages. Firstly, during the **Dialogue Analysis** stage, multiple agents play different roles to extract the user’s psychological state from the dialogue context. Then in the **Strategy Deliberation** stage, we retrieve similar cases from the dataset and integrate them into the following deliberation process to alleviate the preference bias of LLMs. Following this, multiple valid strategies are retained for further utilization. In the **Response Generation** stage, these strategies are used to generate diverse responses, from which the optimal response is selected through multi-agent debate and collaboration. 

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

# MultiAgentESC: A LLM-based Multi-Agent Collaboration Framework for Emotional Support Conversation

As demonstrated in the Figure, MultiAgentESC consists of three stages. Firstly, during the **Dialogue Analysis** stage, multiple agents play different roles to extract the user’s psychological state from the dialogue context. Then in the **Strategy Deliberation** stage, we retrieve similar cases from the dataset and integrate them into the following deliberation process to alleviate the preference bias of LLMs. Following this, multiple valid strategies are retained for further utilization. In the **Response Generation** stage, these strategies are used to generate diverse responses, from which the optimal response is selected through multi-agent debate and collaboration. 

![image](https://github.com/MindIntLab-HFUT/MultiAgentESC/blob/main/image/model4.png)

## Quick Start

#### 1. Clone this project locally
```bash
git clone https://github.com/MindIntLab-HFUT/MultiAgentESC.git
```

#### 2. Navigate to the directory
```bash
cd MultiAgentESC
```

#### 3. Set up the environment
```bash
pip install -r requirements.txt
```

#### 4. Replace the relevant path

Replace paths in parse_args() in main.py

#### 5. Run the Python file run.py
```bash
python main.py
```

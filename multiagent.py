import autogen
import json
import heapq
import re
from autogen import Cache
import numpy as np
from sentence_transformers import util
from collections import Counter


def is_complex(prompt, config_list, cache_path_root):
    '''
    我们认为当对话内容体现出用户当前的情感状态，用户寻求帮助的原因以及用户的意图时，对话足够复杂，需要多个智能体协作完成
    '''
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,    # 
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )

    flag = True if "yes" in response.lower() else False
    return flag


def single_agent_response(prompt, config_list, cache_path_root):
    '''When multi-agent is not needed, we use single agent to generate responses through zero shot.'''
    
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,    # 100
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    
    try: 
        response = re.findall(r'Response:\s*(.*)', response)[0].strip()
    except:
        response = "None"
    return response


def get_emotion(prompt, config_list, cache_path_root): 
    agent = autogen.ConversableAgent(
        name='Emotion Perception Agent',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )

    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        emotion = re.findall(r'Emotion:(.*)', response.split("\n")[0])[0].strip()
    except:
        emotion = "Negative"    # 统一为 "negative"

    return emotion, response


def get_cause(prompt, config_list, cache_path_root):
    '''Cause == Event'''

    agent = autogen.ConversableAgent(
        name='Cause Perception Agent',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        cause = re.findall(r'Event:(.*)', response.split("\n")[0])[0].strip()
    except:
        cause = "Not mention"    # 统一为 "not mention"

    return cause, response
    

def get_intention(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Intention Perception Agent',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        intention = re.findall(r'Intention:(.*)', response.split("\n")[0])[0].strip()
    except:
        intention = "Not mention"    # 统一为 "not mention"

    return intention, response


def select_strategy_by_group(emo_and_reason, cau_and_reason,  int_and_reason, context, examples, config_list, agent_num=3):
    prompt = f'''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and dialogue context, please select a strategy for the 'Assistant' to generate an appropriate response, and explain why. Your strategy should differnet from others as much as possible.
The following are examples of different strategies, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must include the following elements:
Strategy: Strategy for generating an response. The strategy must appear in the examples. Please choose different strategy from others as much as possible.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Strategy: [strategy]
Reasoning: [reasoning]
'''

    # define agent
    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about strategy selection.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(agent_num):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message="You are a psychological counseling expert.",
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 400,    # 400
            },
            human_input_mode='NEVER',
            description="agent_{i} participate in a group discussion about strategy selection."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=agent_num+1,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat, 
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    chat_history = []
    for history in discussion_results.chat_history[1:]:
        chat_history.append(history["content"])

    try:
        strategy = re.findall(r'Strategy:\s*\[?([a-zA-Z ]+)\]?', " ".join(chat_history))    # str 
    except:
        strategy = re.findall(r'\[([a-zA-Z ]+)\]', examples)
        counter = Counter(strategy)
        most_common = counter.most_common(3)
        strategy = [item[0] for item in most_common]

    return strategy


def get_strategy(emo_and_reason, cau_and_reason,  int_and_reason, context, post, quadruple, model, config_list, n=10):
    '''
    ① screen top k (post, [strategy] response) pair (semantic similar)
    ② analysis which are style similar to current context: hard
    ③ get strategy and examples
    '''
    post_embedding = model.encode(post)
    can_embeddings = [np.array(json.loads(q.split('__SEP__')[3]), dtype=np.float32) for q in quadruple]
    similarities = util.pytorch_cos_sim(post_embedding, can_embeddings)[0].tolist()
    top_indices = heapq.nlargest(n, range(len(similarities)), key=lambda i: similarities[i])
    pairs = []
    for i in top_indices:
        post, response, strategy, _  = quadruple[i].split('__SEP__')
        pairs.append((post, f"[{strategy}] {response}"))

    examples = "\n\n".join([f"{pair[0]}\n{pair[1]}" for pair in pairs])
    strategy = select_strategy_by_group(emo_and_reason, cau_and_reason,  int_and_reason, context, examples, config_list)
    return strategy, pairs


def response_with_strategy(context, emo_and_reason, cau_and_reason, int_and_reason, strategy, examples, config_list, cache_path_root):
    '''Only 1 strategy is selected. Generate response directly.'''
    prompt = f'''You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}


Please generate a response from the Assistant's perspective using the {strategy} strategy.
The following are examples of this strategy, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must be fewer than 30 words and must follow this format:
Response: [strategy] [response]
'''
    
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,    # 100
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    
    try: 
        response = re.findall(r'Response:\s*\[[a-zA-Z ]+\](.*)', response)[0].strip()
    except:
        if strategy in response:
            response = response.split(strategy, 1)[1].strip()
        else:
            response = "None"
    
    return response


def debate(context, emo_and_reason, cau_and_reason, int_and_reason, responses, config_list):

    responses_template = "\n\n".join(responses)

    prompt = f'''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and dialogue context, please select the most appropriate response from the following options and explain why. 

### Response
{responses_template}

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [reasoning]'''
    
    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about which response is the most appropriate.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(len(responses)):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message=f'''You are a psychologist who is good at listening to others' opinions and reflecting on your own thoughts. You are currently participating in a group discussion about which response is the most appropriate, and you are inclined to support the response "{responses[i]}". However, during the discussion, you need to carefully consider others' perspectives and reflect on your own viewpoint, ultimately reaching a reliable answer.''',
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 400,    # 400
            },
            human_input_mode='NEVER',
            description="agent_{i} participate in a group discussion about which response is the most appropriate."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=len(responses)+1,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat, 
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    chat_history = []
    for history in discussion_results.chat_history[1:]:
        chat_history.append(history["content"])
    
    return chat_history


def reflect(context, emo_and_reason, cau_and_reason, int_and_reason, debate_history, responses, config_list):
    discussion_content = "\n\n".join(debate_history)
    
    prompt = f'''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and the context of the dialogue, a group discussion is taking place to determine which response is the most appropriate.

### Discussion content
{discussion_content}

You should carefully analyze the various different viewpoints above, reflect on your own thoughts, and ultimately arrive at a convincing result. Your thought can be changed if you believe the viewpoints of others are more reasonable.

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [reasoning]'''
    
    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about which response is the most appropriate.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(len(responses)):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message=f'''You are a psychologist who is good at listening to others' opinions and reflecting on your own thoughts. You are currently participating in a group discussion about which response is the most appropriate, and you are inclined to support the response "{responses[i]}". However, during the discussion, you need to carefully consider others' perspectives and reflect on your own viewpoint, ultimately reaching a reliable answer. Your thought can be changed if you believe the viewpoints of others are more reasonable.''',
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 400,    # 400
            },
            human_input_mode='NEVER',
            description="agent_{i} participate in a group discussion about which response is the most appropriate."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=len(responses)+1,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat, 
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    reflection_history = []
    for history in discussion_results.chat_history[1:]:
        reflection_history.append(history["content"])
    
    return reflection_history


def vote(results):
    count = {}
    strat2response = {}
    for result in results:
        try:
            response, _ = result.split("\nReasoning")
            strategy, response = re.findall(r'Response:\s*\[([a-zA-Z ]+)\](.*)', response)[0]
            strategy, response = strategy.strip(), response.strip()
            if strategy not in count:
                count[strategy] = 0
            count[strategy] += 1
            if strategy not in strat2response:
                strat2response[strategy] = response
        except:
            continue
    if len(count) == 0:
        return ["None"], ["None"]
    max_count = max(count.values())
    max_strat = [key for key, value in count.items() if value == max_count]
    responses = [strat2response[strat] for strat in max_strat]

    return max_strat, responses
        

def judge(context, strategies, responses, config_list, cache_path_root):
    template = []
    for strategy, response in zip(strategies, responses):
        template.append(f"[{strategy}] {response}")
    template_responses = "\n\n".join(template)

    prompt = f'''You will be provided with a dialogue context between an 'Assistant' and a 'User'. 

### Dialogue context
{context}

The following are responses generated by the therapist using different strategies, all presented in the format of <[strategy] response>. Please select the most appropriate response and explain why.

### Examples
{template_responses}

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [reasoning]'''
    
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    
    try: 
        strategy, response = re.findall(r'Response:\s*\[([a-zA-Z ]+)\](.*)', response)[0]
    except:
        strategy, response = "None", "None"
    
    return strategy, response


def self_reflection(context, pred_strategy, response, config_list, cache_path_root):

    prompt = f'''You will be provided with a dialogue context between an 'Assistant' and a 'User'. 

### Dialogue context
{context}

The following is a responses generated by the therapist using {pred_strategy} strategy, presented in the format of <[strategy] response>. Please analyze whether this response is consistent with the ongoing conversation, whether it aligns with the strategy, and whether it effectively helps alleviate the user's emotional stress.

### Response
[{pred_strategy}] {response}

If the respones meets the above requirements, please return it as is; if not, please modify the response and provide a more refined version. Refined version must less than 30 words.

Your answer must include the following elements:
Response: original response or refined response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Response: [strategy] [origianl/refined response]
Reasoning: [reasoning]'''
    
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 400,    # 400
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[
                {
                    'content': prompt, 
                    'role': 'user'
                }
            ],
            cache=cache,
        )
    
    try: 
        strategy, response = re.findall(r'Response:\s*\[([a-zA-Z ]+)\](.*)', response)[0]
    except:
        strategy, response = "None", "None"
    
    return strategy.strip(), response.strip()

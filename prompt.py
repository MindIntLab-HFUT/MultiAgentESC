prompts = {
    "behavior_control": '''### Instruction
You are a psychological counseling expert. You will be provided with an incomplete conversation between an Assistant and a User. 
Please analyze whether this conversation reflects the user's current emotional state, the reason the user is seeking emotional support, and how the user plans to cope with the event. 
If all three points are reflected, please reply "YES," otherwise reply "NO." 

### Conversation
{context}

Your answer must include two parts:
1. "YES" or "NO"
2. If "YES", briefly explain how the conversation reflects these elements; if "NO", explain which elements are missing.

Your answer must follow this format:
1. [YES or NO]
2. [explaination]
''',

    "zero_shot": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context. 

### Dialogue context
{context}

Your answer must be fewer than 30 words and must follow this format:
Response: [response]
''',

    "get_emotion": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Please infer the emotional state expressed in the user's last utterance.

### Dialogue context
{context}

Your answer must include the following elements:
Emotion: the emotion user expressed in their last utterance.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Emotion: [emotion]
Reasoning: [reasoning]
''',

    "get_cause": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Another agent analyzes the conversation and infers the emotional state expressed by the user in their last utterance. 

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

Please infer the specific event that led to the user's emotional state based on the dialogue context. Your answer must include the following elements:
Event: the specific event that led to the user's emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Event: [event]
Reasoning: [reasoning]
''',

    "get_intention": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, infering the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state. 

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

Please reasonably infer the user's intention based on the dialogue context, with the goal of addressing the event that lead to their emotional state. Your answer must include the following elements:
Intention: user's intention which aims to address the event that lead to their emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format: 
Intention: [intention]
Reasoning: [reasoning]
'''
}

def get_prompt(prompt_name):
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found.")
    return prompts[prompt_name]
import json
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import torch
import autogen
import argparse
from strategy import strategy_definitions
from prompt import get_prompt
from multiagent import (
    is_complex, 
    single_agent_response,
    get_emotion,
    get_cause,
    get_intention,
    get_strategy,
    response_with_strategy,
    debate,
    reflect,
    vote,
    judge,
    self_reflection
)


def get_embeddings(model, targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    targets = [target[0] for target in targets]
    embeddings = model.encode(targets).tolist()
    return embeddings


def remove_quotes(s):
    # Check if the string starts and ends with a quote
    if s.startswith('"'):
        s = s[1:]
    if s.endswith('"'):
        s = s[:-1] 
    return s  


def get_cases():
    with open('./dataset/ESConv.json', 'r') as f:
        samples = json.load(f)
        random.seed(42)
        random.shuffle(samples)     

        cases = samples[400:420]    # 选取20个对话作为cases
        ret_cases = []
        # 生成固定的格式 
        for case in cases:
            history = ""
            _num = 0
            for e in case['dialog']:
                if _num ==0 or e['speaker'] == "seeker":
                    history += f"User: {e['content'].strip()} " if e['speaker'] == "seeker" else f"Assistant: {e['content'].strip()} "
                else:
                    ret_cases.append({
                        "history": history,
                        "response": e['content'].strip(),
                        "strategy": e['annotation']['strategy']
                    })
                    history += f"Assistant: {e['content'].strip()} "
                _num += 1
        return ret_cases

                 
def clean_strategy(strategy):
    cleaned_strategy = set()
    for s in strategy:
        if s.lower() == "question":
            cleaned_strategy.add("Question")
        elif s.lower() == "restatement or paraphrasing":
            cleaned_strategy.add("Restatement or Paraphrasing")
        elif s.lower() == "reflection of feelings":
            cleaned_strategy.add("Reflection of feelings")
        elif s.lower() == "self-disclosure":
            cleaned_strategy.add("Self-disclosure")
        elif s.lower() == "affirmation and reassurance":
            cleaned_strategy.add("Affirmation and Reassurance")
        elif s.lower() == "providing suggestions":
            cleaned_strategy.add("Providing Suggestions")
        elif s.lower() == "information":
            cleaned_strategy.add("Information")
        elif s.lower() == "others":
            cleaned_strategy.add("Others")    
    return list(cleaned_strategy)


def parse_args():
    parser = argparse.ArgumentParser()
    # Adding arguments for each of the variables
    parser.add_argument("--seed", type=int, default=42)                         
    parser.add_argument("--dataset", type=str, default="dataset/ESConv.json")                                     
    parser.add_argument("--model_path", type=str, default="all-roberta-large-v1")                                  
    parser.add_argument("--llm_name", type=str, default="qwen2.5:32b")                                  
    parser.add_argument("--cache_path_root", type=str, default="")                         
    parser.add_argument("--save_path", type=str, default="")
   
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_quadruple(model, dataset):  
    path = "./embeddings.txt"
    if not os.path.exists(path):
        with open(path, 'a') as txt:
            for sample in tqdm(dataset[100:]):
                targets = []
                dialog = sample['dialog']   
                for count in range(len(dialog)-1):
                    if dialog[count]['speaker'] == "seeker" and dialog[count+1]['speaker'] == "supporter":
                        targets.append((dialog[count]['content'].strip(), dialog[count+1]['content'].strip(), dialog[count+1]['annotation']['strategy']))
                embeddings = get_embeddings(model, targets)
                for triple, embedding in zip(targets, embeddings):
                    post = triple[0]
                    response = triple[1]
                    strategy = triple[2]
                    line = f"{post}__SEP__{response}__SEP__{strategy}__SEP__{embedding}".replace("\n", "\\n") + "\n"
                    txt.write(line)
        with open(path, "r") as txt:
            quadruple = txt.readlines()
    else:
        with open(path, "r") as txt:
            quadruple = txt.readlines()

    return quadruple


def json2natural(history):
    natural_language = ""
    for u in history:
        content = u["content"].strip()
        role = u["role"].capitalize() if "role" in u.keys() else u["speaker"].capitalize()
        if role == "Supporter":
            role = "Assistant"
        if role == "Seeker":
            role = "User"
        
        natural_language += f"{role}: {content} "
    return natural_language.strip()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    samples = dataset[:100]     
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=".",
        filter_dict={
            "model": [args.llm_name],
        }
    )
    
    model = SentenceTransformer(args.model_path)
    quadruple = get_quadruple(model, dataset)
 
    ret = []
    for sample in tqdm(samples):
        dialog = sample['dialog']   
        count = 0   
        history = []    
        
        while True:
            save = {}
            if count == len(dialog):   
                break
            
            if count != 0 and dialog[count]["speaker"] == "supporter":    
            
                if (count < len(dialog) -1 and dialog[count+1]['speaker'] != "supporter") or (count == len(dialog)-1):    
                    '''
                    need not combine utterance!
                    operations:
                        1. strategy --->  STRATEGY
                        2. reference ---> REFERENCE
                        3. context ---> CONTEXT
                        4. history, count changes.
                    '''
                    save["strategy"] = dialog[count]['annotation']['strategy']
                    save["reference"] = dialog[count]['content'].strip()  

                    context = json2natural(history)    
                    save["context"] = context
                    post = history[-1]['content']  

                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"]=="seeker" else "assistant"
                    })    
                    count += 1    

                elif count < len(dialog) -1 and dialog[count+1]['speaker'] == "supporter":
                    '''
                    need combine utterance!
                    operations:
                        1. strategy --->  STRATEGY
                        2. reference ---> REFERENCE
                        3. context ---> CONTEXT
                        4. history, count changes.
                    '''
                    
                    save["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count+1]['annotation']['strategy']}"   
                    save["reference"] = dialog[count]['content'].strip() + ' ' + dialog[count+1]['content'].strip()    

                    context = json2natural(history)    
                    save["context"] = context
                    post = history[-1]['content']  

                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"]=="seeker" else "assistant"
                    })
                    history.append({
                        "content": dialog[count+1]["content"].strip(),
                        "role": "user" if dialog[count+1]["speaker"]=="seeker" else "assistant"
                    })

                    count += 2    # count + 2
                                

                if count <= 5 or not is_complex(get_prompt("behavior_control").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root):    # 处于对话的开始阶段，我们认为不需要有策略的约束，也不需要多个agents协作
                    response = single_agent_response(get_prompt("zero_shot").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root)
                    save["response"] = response
                    save["pred_strategy"] = "None"

                else:
                    emotion, emo_and_reason = get_emotion(get_prompt("get_emotion").format(context=context), config_list, args.cache_path_root)
                    cause, cau_and_reason = get_cause(get_prompt("get_cause").format(emo_and_reason=emo_and_reason, context=context), config_list, args.cache_path_root)
                    intention, int_and_reason = get_intention(get_prompt("get_intention").format(emo_and_reason=emo_and_reason, cau_and_reason=cau_and_reason, context=context), config_list, args.cache_path_root)
                    pred_strategy, pairs = get_strategy(emo_and_reason, cau_and_reason, int_and_reason, context, post, quadruple, model, config_list)
                    pred_strategy = clean_strategy(pred_strategy)    # delete unqualified strategies
                    save["emotion"], save["cause"], save["intention"]  = emotion, cause, intention
                    if len(pred_strategy) == 0:
                        response = single_agent_response(get_prompt("zero_shot").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root)
                        save["response"] = response
                        save["pred_strategy"] = "None"
                    else:
                        if len(pred_strategy) == 1:    # 很少
                            examples = ""
                            for pair in pairs:
                                strat = pair[1].split("]", 1)[0].strip("[").strip()
                                if strat == pred_strategy[0]:
                                    examples += f"{pair[0]}\n{pair[1]}\n\n"   
                            examples = examples.strip()   
                            response = response_with_strategy(context, emo_and_reason, cau_and_reason, int_and_reason, pred_strategy[0], examples, config_list, args.cache_path_root)
                            pred_strategy = pred_strategy[0]
                            save["ori_response"], save["pred_strategy"] = response, pred_strategy
                        else:
                            responses = []
                            for strat in pred_strategy:
                                examples = ""
                                for pair in pairs:
                                    p_strat = pair[1].split("]", 1)[0].strip("[").strip()
                                    if p_strat == strat:
                                        examples += f"{pair[0]}\n{pair[1]}\n\n"   
                                examples = examples.strip()   
                                response = response_with_strategy(context, emo_and_reason, cau_and_reason, int_and_reason, strat, examples, config_list, args.cache_path_root)
                                responses.append(f'''[{strat}] {response}''')
                            
                            debate_history = debate(context, emo_and_reason, cau_and_reason, int_and_reason, responses, config_list)
                            reflection_result = reflect(context, emo_and_reason, cau_and_reason, int_and_reason, debate_history, responses, config_list)
                            strats, responses = vote(reflection_result)

                            if len(strats)==1 and strats[0] != "None":   ## vote
                                pred_strategy, response = strats[0].strip(), responses[0].strip()
                                save["ori_response"], save["pred_strategy"] = response, pred_strategy
                            else:
                                pred_strategy, response = judge(context, strats, responses, config_list, args.cache_path_root)
                                save["ori_response"], save["pred_strategy"] = response, pred_strategy

                        _, response = self_reflection(context, pred_strategy, response, config_list, args.cache_path_root)
                        save["response"] = response
                                                    
                ret.append(save)
                
            else:    
                
                history.append({
                    "content": dialog[count]["content"].strip(),
                    "role": "user" if dialog[count]["speaker"]=="seeker" else "assistant"
                })
                count += 1

    with open(args.save_path, 'w') as f:
        json.dump(ret, f, indent=4)
            
            
           
            
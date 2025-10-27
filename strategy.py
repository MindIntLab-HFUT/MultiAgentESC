strategy = {
    "Question": "Asking for information related to the problem to help the user articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.",
    "Restatement or Paraphrasing": "A simple, more concise rephrasing of the user's statements that could help them see their situation more clearly.",
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the user to express your empathy.", 
    "Affirmation and Reassurance": "Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement.",
    "Providing Suggestions": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do.",
    "Information": "Provide useful information to the user, for example with data, facts, opinions, resources, or by answering questions.",
    "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories."
}

strategy_definitions = "Here are 8 strategies for generating responses: \n\n"
for key, value in strategy.items():
    strategy_definitions += f"{key}: {value}\n"


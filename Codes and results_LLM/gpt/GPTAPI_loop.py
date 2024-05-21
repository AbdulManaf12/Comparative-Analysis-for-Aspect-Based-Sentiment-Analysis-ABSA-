"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""
import re
import time
import pandas as pd
import os
import openai
import os
import csv
import time
from itertools import islice


# initialize OpenAI API key
openai.api_key = ""
def analyze(text):
    prompt_aspect = f"""As an aspect-based sentiment analyzer, your task is to extract aspects and corresponding sentiments (Positive, Negative, or Neutral) from people's reviews. Each review consists of a text and may contain multiple aspects and their corresponding sentiments. Your goal is to process the reviews and extract the aspects and sentiments from them.

Please analyze the provided review below and present the aspects and sentiments in the specified format. Ensure that the aspects you provide show some relation and limit them to the most important aspects in the review. Please do not explain any aspect or sentiment. I do not need any extra text. Just the lists of aspects and the sentiments. 

Review: {text}

Required output Format: 

aspect: ['aspect1', 'aspect2', 'aspect3', ...]
Sentiment: ['sentiment1', 'sentiment2', 'sentiment3', ..]"""

    result_score_turbo = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2, 
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt_aspect}
        ] 
    )
  
    score_turbo = result_score_turbo.choices[0].message.content.strip()
    return score_turbo

filename="16res_test"
if(os.path.isfile("results/ASTE-Data-V2/"+filename+"_gpt.csv")):
    temp = pd.read_csv("results/ASTE-Data-V2/"+filename+"_gpt.csv")
else:
    temp=[]
print(len(temp))
df = pd.read_csv("C:/Users/nimra/OneDrive/ABSA work/codes/16res_test_original.csv")
df = df.dropna(axis=1, how='all')
print(len(df))
df.head(2)

# for i in range(len(temp),len(df)):
# Iterate over the rows in the data
feedback=[]
keywords= ['aspect', 'sentiment']
results = {rubric: [] for rubric in keywords}
for i in range(len(temp), 50):
  data = df[i:i+1]
  for index, row in data.iterrows():
    # Get the text to assess
    input = row["text"]    

    response = analyze(input)
    print(response)
    print(type(response))
    feedback.append(str(response))   
    print(len(feedback)) 
    for i in range(len(keywords)):
        keyword = keywords[i]
        pattern = re.compile(rf"(?i)\b{keyword}\b(.+?)(?=\b{keywords[i+1]}\b|$)" if i < len(keywords) - 1 else rf"(?i)\b{keyword}\b(.+?$)", re.DOTALL)
        match = pattern.search(response)
        if match:
            results[keyword].append(match.group(1).strip())
        else:
            results[keyword].append("")
  data['Response_gpt'] = feedback
  print(results['sentiment'])
  for keyword in keywords:
      data[keyword+"_gpt"] = results[keyword]        
  print(response)
  print(row)
#   data.to_csv(filename+"_gpt.csv", index = False,  header=False, mode='a')
  data.to_csv("results/ASTE-Data-V2/"+filename+"_gpt.csv", index = False, mode='a', header=False)
  results = {rubric: [] for rubric in keywords}
  feedback=[]
  time.sleep(20)    

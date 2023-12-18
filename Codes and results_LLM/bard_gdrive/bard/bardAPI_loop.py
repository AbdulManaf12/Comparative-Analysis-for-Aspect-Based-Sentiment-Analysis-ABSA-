"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""
import re
import time
import google.generativeai as palm
import pandas as pd
import pandas as pd
import os
palm.configure(api_key="AIzaSyBrNo4zVBIAdy1x0YnO6ElzV3UKjOjCI9s")

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.2,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
}
path = "results/ASTE-Data-V2/"
filename="16res_test"
if(os.path.isfile(path+filename+"_bard.csv")):
    temp = pd.read_csv(path+filename+"_bard.csv")
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
for i in range(len(temp), len(df)):
  data = df[i:i+1]
  for index, row in data.iterrows():
    # Get the text to assess
    input = row["text"]    
    # prompt = f"""input: Input1: 
    # Extremely slow service and lack of organized . Even there are tables available , customers still need to wait for an hour .
    # Input2: 
    # Had the pleasure of a memorable dining experience here last night . Schnitzel was the entree of choice . The entree was perfect , the staff warm . Yes , we will return .

    # output: Output1: 
    # Aspect: 'service'
    # Sentiment: 'Negative'
    # Output2:
    # Aspect: 'entree', 'staff'
    # Sentiment: 'Positive', 'Positive'
    # input: {input}
    # output:"""
    prompt = f"""As an aspect-based sentiment analyzer, your task is to extract aspects and corresponding sentiments (Positive, Negative, or Neutral) from people's reviews. Each review consists of a text and may contain multiple aspects and their corresponding sentiments. Your goal is to process the reviews and extract the aspects and sentiments from them.

Please analyze the provided review below and present the aspects and sentiments in the specified format. Ensure that the aspects you provide show some relation and limit them to the most important aspects in the review. Please do not explain any aspect or sentiment. I do not need any extra text. Just the lists of aspects and the sentiments. 

Review: {input}

Required output Format: 

aspect: ['aspect1', 'aspect2', 'aspect3', ...]
Sentiment: ['sentiment1', 'sentiment2', 'sentiment3', ..]"""
    response = palm.generate_text(
      **defaults,
      prompt=prompt
    )
    print(response.result)
    print(type(response.result))
    feedback.append(str(response.result))   
    print(len(feedback)) 
    for i in range(len(keywords)):
        keyword = keywords[i]
        pattern = re.compile(rf"(?i)\b{keyword}\b(.+?)(?=\b{keywords[i+1]}\b|$)" if i < len(keywords) - 1 else rf"(?i)\b{keyword}\b(.+?$)", re.DOTALL)
        match = pattern.search(response.result)
        if match:
            results[keyword].append(match.group(1).strip())
        else:
            results[keyword].append("")
  data['Response_gpt'] = feedback
  print(results['sentiment'])
  for keyword in keywords:
      data[keyword+"_bard"] = results[keyword]        
  print(response.result)
  print(row)
  data.to_csv(path+filename+"_bard.csv", index = False,  header=False, mode='a')
  results = {rubric: [] for rubric in keywords}
  feedback=[]
  time.sleep(20)    

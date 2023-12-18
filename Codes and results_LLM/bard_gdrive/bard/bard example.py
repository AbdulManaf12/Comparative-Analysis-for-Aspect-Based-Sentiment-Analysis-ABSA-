"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""

import google.generativeai as palm

palm.configure(api_key="AIzaSyBrNo4zVBIAdy1x0YnO6ElzV3UKjOjCI9s")

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.7,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
}
input = "Fifty Licks has a fun variety of flavors of ice cream to choose from , and you can tell that it\ 's quality creamy ice cream . Some of the flavor names are funny like Chocolate as @ ! # $ @ '' and `` Vanilla as @ # $ # @ # '' . It\ 's funny to hear people filling in the blanks when they order . When you come in , there is someone to greet you and give you a chance to sample any of the flavors before making your decision . They were all so good that it was hard for me to decide , but I did end up going with the Thai Rice ice cream ( creamy jasmine rice pudding infused with Pandan ) . Where else can you get something like that ? They do have truly unique flavors and the staff is there to help you make your decision . I also liked the fact that there wasn\'t a very long line for me to wait in to order . It\ 's a small cozy place with seating inside and out . Or you can take your ice cream to go as you wander the area . Keep in mind that the ice cream is a little on the expensive side , but it could be justifiable with their unique and delicious flavors ."
prompt = f"""input: Input1: 
Extremely slow service and lack of organized . Even there are tables available , customers still need to wait for an hour .
Input2: 
Had the pleasure of a memorable dining experience here last night . Schnitzel was the entree of choice . The entree was perfect , the staff warm . Yes , we will return .

output: Output1: 
Aspect: [service]
Sentiment: [Negative]
Output2:
Aspect: [entree, staff]
Sentiment: [Positive, Positive]
input: {input}
output:"""

response = palm.generate_text(
  **defaults,
  prompt=prompt
)
print(response.result)
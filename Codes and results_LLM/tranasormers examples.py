import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import ast
# Function to convert the string representation of lists to actual lists
def convert_to_list(row):
    return ast.literal_eval(row)

# # Sample data
# data = =pd.read_csv() pd.DataFrame({
#     'text': ["Serves really good sushi .",
#              "Not the biggest portions but adequate .",
#              "Green Tea creme brulee is a must !",
#              "It has great sushi and even better service .",
#              "The entire staff was extremely accomodating and tended to my every need .",
#              "The owner is belligerent to guests that have a complaint ."],
#     'aspect': [['sushi'], ['portions', 'portions'], ['Green Tea creme brulee'], ['sushi', 'service'], ['staff'], ['owner']],
#     'sentiment': [['POS'], ['NEU', 'NEU'], ['POS'], ['POS', 'POS'], ['POS'], ['NEG']]
# })
data = pd.read_csv("processed_ASTE-Data-V2/16res_train_original.csv")
# Apply the converter function to the 'Aspect_Category' column
data['aspect'] = data['aspect'].apply(convert_to_list)
data['sentiment'] = data['sentiment'].apply(convert_to_list)

# Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text and aspects
inputs = tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
aspect_embeddings = model(**inputs)['last_hidden_state']

# Convert sentiment labels to numerical representations
sentiments = {'POS': 0, 'NEU': 1, 'NEG': 2}
data['sentiment'] = data['sentiment'].apply(lambda x: [sentiments[s] for s in x])

# Flatten the aspect and sentiment lists for multi-label classification
data_flat = data.explode('aspect').explode('sentiment').reset_index(drop=True)

# Prepare training and testing data
X_train, X_test, y_train, y_test = train_test_split(aspect_embeddings, data_flat['sentiment'], test_size=0.2, random_state=42)

# Train an SVM classifier
classifier = SVC()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict the sentiment for new inputs
new_text = ["The sushi was amazing but the service was slow."]
new_aspect = [['sushi', 'service']]

new_inputs = tokenizer(new_text, padding=True, truncation=True, return_tensors="pt")
new_aspect_embeddings = model(**new_inputs)['last_hidden_state']

new_pred = classifier.predict(new_aspect_embeddings)
print(f"Predicted sentiment: {new_pred}")


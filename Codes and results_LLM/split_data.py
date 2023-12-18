import pandas as pd
from sklearn.model_selection import train_test_split
path = "C:/Users/Nimra/OneDrive/ABSA work/Datasets/DOTSA Reviews Dataset/"
filename="new_Clothing"
# Load the CSV file into a pandas DataFrame
data = pd.read_csv(path+filename+".csv")
new_path = "processed_dotsa/"
# Split the data into train (60%), test (20%), and validation (20%) sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
test_data, validation_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the three sets into separate CSV files
train_data.to_csv(new_path+filename+'_train.csv', index=False)
test_data.to_csv(new_path+filename+'_test.csv', index=False)
validation_data.to_csv(new_path+filename+'_val.csv', index=False)
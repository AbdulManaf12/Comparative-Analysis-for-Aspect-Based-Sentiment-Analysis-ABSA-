import ast

data_str = "[([17], [16], 'POS'), ([19], [16], 'POS')]"

def extract_triplets(data_str):
    data_list = ast.literal_eval(data_str)
    triplets = []
    for triplet in data_list:
        triplets.append(triplet)
    return triplets

extracted_triplets = extract_triplets(data_str)
for idx, triplet in enumerate(extracted_triplets, start=1):
    print(f"Triplet {idx}: {triplet}")
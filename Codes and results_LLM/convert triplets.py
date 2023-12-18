import csv
import re
import ast
org_path = ""
path = "processed_ASTE-Data-V2/"
filename="16res_train"
def extract_triplets(data_str):
    data_list = ast.literal_eval(data_str)
    triplets = []
    for triplet in data_list:
        triplets.append(triplet)
    return triplets
def extract_words_by_indexes(text, indexes):
    words = text.split()
    extracted_words = [words[idx] for idx in indexes]
    return " ".join(extracted_words)

def create_csv_file(examples_file):
    with open(examples_file, "r") as f:
        examples = []
        for line in f:
            example = line.strip("\n").split("####")
            print(example)
            if len(example) == 2:
                text = example[0]
                # print(text)
                aspects = []
                opinions=[]
                sentiments=[]
                for triplet in extract_triplets(example[1]):
                    # print(text)
                    aspects_indexes= triplet[0]
                    # print(aspects_indexes)
                    aspects_words = extract_words_by_indexes(text, aspects_indexes)
                    opinion_indexes= triplet[1]
                    # print(opinion_indexes)
                    opinion_words = extract_words_by_indexes(text, opinion_indexes)
                    # print(opinion_words)
                    # triplet=triplet[3:]
                    # triplet = re.sub("[\[]", "", triplet).replace("(", "")
                    # triplet = re.sub("[\[()]", "", triplet)
                    # triplet = re.sub("[\[]", "", text).replace(")", "")
                    
                    # print(triplet)
                    # aspect = triplet.strip().split("], ")[0]
                    # print(aspect)
                    sentiment = triplet[2]
                    print(sentiment)
                    sentiments.append(sentiment)
                    aspects.append(aspects_words)
                    opinions.append(opinion_words)

                print("\n\n")
                print(aspects)
                print(opinions)
                
                examples.append((text, aspects, opinions, sentiments))
            # print(examples)
    with open(path+filename+"_original.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["text", "aspect", "opinion", "sentiment"])
        for example in examples:
            text = example[0]
            aspect = example[1]
            opinion = example[2]
            sentiment = example[3]
                       
            writer.writerow([text, aspect, opinion, sentiment])

if __name__ == "__main__":
    examples_file = "C:/Users/Nimra/OneDrive/ABSA work/Datasets/SemEval-Triplet-data-master/ASTE-Data-V2-EMNLP2020/16res/train_triplets.txt"
    print(len(examples_file))
    create_csv_file(examples_file)

import openai
import os
import csv
import time
from itertools import islice

# initialize OpenAI API key
openai.api_key = ""


prompt1 = '''

You are working as an aspect analyzer and sentiment analyzer. you need to extract aspects and sentiments from people reviews. Each review consists of a text and may contain multiple aspects and their corresponding sentiments. As an aspect analyzer, you need to process reviews and extract the aspects and sentiments from it. For your help  I am providing you some reviews, and its aspect and sentiments. 


1. I would recommend these sellers . They were quick about getting books out . the books were all in shape described . very very satisfied with their service and look for them in the future for other purchases .	
Aspects: books, books--service, books--sellers	
Sentiments: Positive, Positive, Positive

2. this being a true story..it was sad . To imagine a little child going thru that is horrendous , but I kept reading . The ending was good also and it does not leave you hanging .	
Aspects: story, ending	
Sentiments: Mixed, Positive

3. This is a horrific unveiling of the worst kind of abuse told by a most amazingly unselfish person . You never doubt the veracity of the tale . I have no doubt the author downplayed the physical and emotional toll on her during the time she fostered Jodie . Easy to follow , well written and a pleasure to read despite the subject matter . It is nice to see that there are people who care in a world that can be so dreadful .

Aspects: person, subject, It
Sentiments: Positive, Negative, Positive

4. I love this and wear it all the time . The only issue is that it has begun to pill with wear . I have similar ones which I bought in the late 1980\ 's which have never pilled at all . Oh well . Very warm , very convenient for carrying the camera when out with the dogs .

Aspects: similar ones, this
Sentiments: Positive, Mixed

5. nice shirt but way to small . xl was at best 3xl on the small side..
Aspects: shirt, 3xl, xl
Sentiments: Mixed, Negative, Positive

6. I love the shirt and would be glad to order more in the future but , the seam came loose on one of the sleeves already . that 's a bit too fast I would think .
Aspects: shirt, shirt--seam
Sentiments: Mixed, Negative

7. It is a one room apartment , with a kitchen that is not fully equiped . Far from downtown but well comunicated . Mark is very helpful the arrival is very simple .
Aspects: apartment--kitchen, apartment--Mark, apartment, apartment--arrival
Sentiments:  Negative, Positive, Mixed, Positive

8. Bare bones room to stay in while visiting Boston . They offer a private entrance and easily walkable to the transit system . We did n't hang too much in the neighborhood but found it fairly easy to commute from . The Uber ride will run you about 20- to get to Back Bay or Beacon Hill . The room offers a full size bed , refrigerator , kitchen sink and microwave . The Bathroom is quite tiny offering a small standing only shower ( if you 're 200lbs and up , it will be a challenge ) . My husband and I we 're not able to both fit in the bathroom to get ready , so I had to blow dry my hair and do my makeup on the bed with a small mirror . Overall , if you 're looking for a bed and a affordable room , this might be your place . We just wished we would have spent the extra 50- to stay in the center of town with a bigger bed .

Aspects: room--bathroom--standing, neighborhood, room--bathroom, private entrance
Sentiments: Negative, Positive, Negative, Positive

9. Mark was a great host , he was so flexible when our check in times changed . The studio is small , but highly functional .
Aspects: studio, studio--Mark
Sentiments: Mixed, Positive

10. I\ 'll start by admitting that we probably should have picked a different restaurant to eat at for an evening where we just had a couple hours to enjoy . However , we had a scoutmob and thought we\ 'd try the charming-looking restaurant just walking distance from our apartment . We waited at the front of the restaurant for several minutes before a server finally ushered us to the bar in the back . The owner then shuffled us to a corner of the bar and said the wait was 5-10 minutes . '' We waited for an hour . I don\'t mind waiting , I just need to be given an appropriate estimate . we really didn\'t have an hour to wait that night , and would have come back a different evening instead . Once we were finally seated , the owner dropped by our table several times and delivered our food personally . While this was a nice touch , he seemed more concerned with this over-the-top type of service then on swift and efficient service . I don\'t need a napkin personally dropped in my lap for me , I just want you to bring me what I ordered in a reasonable amount of time . ( fyi- we never received the wine we ordered , though they did try to charge us for it ) . The food itself was delicious . Still it took forever to get our bill and when we finally did , it was wrong . They were good about fixing this , but returned the check with 18  gratuity ( of the original total bill -- including drinks we never got ) included.. We weren\'t about to stiff them , but it also seems unfair to add such an outrageous tip to a small table -- especially when their service was so very poor . Service : overly-attentive , inefficient , slow . Food : delicious ( awesome gnocchi ) . Bill : believe what it says here . Whether they are deceitful or just plain sloppy , its questionable . they\ 'd get just one star if it weren\'t that the gnocchi was so yummy .

Aspects: restaurant--owner, restaurant--service, restaurant--food, restaurant--gnocchi, restaurant--Service, restaurant, restaurant--Food, restaurant--touch, bill, They

Sentiments: Positive, Negative, Positive, Positive, Negative, Mixed, Positive, Positive, Negative, Positive

11. We ordered shrimp ... fried rice and cod n chips ...  . cheesecake for dessert . Was delivered on time ... was hot and portions were quite large and prices are good ! We will be ordering again ... ..next time Pizza ! ! !
Aspects: fried rice, dessert--cheesecake--portions, shrimp--prices, cod n chips--prices, fried rice--prices, dessert--cheesecake--prices, shrimp--portions, cod n chips--portions, fried rice--portions, Pizza

Sentiments: Positive, Positive, Positive, Positive, Positive, Positive, Positive, Positive, Positive, Positive

12. I am very disappointed with my recent visit . When this first opened , I gave it 5 stars . But the quality and the taste of the food has deteriorated dramatically since then . All the curries taste very similar : tomatoe-y and creamy . The chicken in the butter chicken seemed to be left over from something else , the malai kofta had a bad smell like they were very old or had gone bad ( they replaced it and the new batch tasted a bit better ) , and the vindaloo was just average . Even the paneer pakora tasted like the paneer had gone bad . The staff was very polite and tried to be as accommodating as possible , but the food was what it was . I am very disappointed . This used to be a good neighborhood Indian go-to .

Aspects: butter chicken--chicken, new batch, paneer, malai kofta, food, staff, it
Sentiments: Negative, Positive, Negative, Negative, Negative, Positive, Positive
 
Now, as I gave you some examples . Your task is to analyze below provided given review and present the aspects and sentiments. Prove sentiment of every aspect. There is one thing I want you to remember, give aspects which show some relation! also try to limit it to the aspects which are most important in a review. In one line give aspects, and in other line give sentiments.
'''





def analyze(text):
    prompt_aspect = f"{prompt1} 1. {text}"
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

def main():
    csv_file_path = "Reviews Dataset/new_Books.csv"  # Replace with the actual path to your CSV file
    output_csv_file = "Reviews Dataset/output.csv"  # Replace with the desired path for the output CSV file

    # Check if the file exists
    if not os.path.isfile(csv_file_path):
        print("CSV file not found.")
        return

    # Initialize an empty list to store the results
    results = []

    # Read the first 6 rows from the CSV file and analyze each 'text' column
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in islice(reader, 6):  # Use islice to limit the loop to the first 6 rows
            text = row['Text']
            score_turbo = analyze(text)
            print(score_turbo)

            # Extract aspects and sentiments from the API response
            aspects, sentiments = [], []
            for line in score_turbo.splitlines():
                if line.startswith("Aspects:"):
                    aspects = line.split(":")[1].strip().split(", ")
                elif line.startswith("Sentiments:"):
                    sentiments = line.split(":")[1].strip().split(", ")

            # Create a dictionary with the 'Text', 'Aspects', and 'Sentiments' columns
            result_dict = {'Text': text, 'Aspects': aspects, 'Sentiments': sentiments}
            results.append(result_dict)

            # Wait for 20 seconds between each API call (3 requests per minute)
            time.sleep(20)

    # Write the results to a new CSV file
    with open(output_csv_file, 'u', newline='') as csvfile:
        fieldnames = ['Text', 'Aspects', 'Sentiments']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()

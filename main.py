import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
import nltk
# import editdistance # import is never used, we use nltk.edit_distance
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')

dictionary = {
  0: "ack", 
  1: "affirm", 
  2: "bye", 
  3: "confirm", 
  4: "deny", 
  5: "hello", 
  6: "inform", 
  7: "negate", 
  8: "null",
  9: "repeat", 
  10: "reqalts", 
  11: "reqmore", 
  12: "request", 
  13: "restart", 
  14: "thankyou"}


def preprocess(sentence):
    processed_sentence = []
    # preprocessing
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # remove punctuation
    stemmer = PorterStemmer()
    tokenized = tokenizer.tokenize(sentence)

    stemmed = []
    for token in tokenized:
        # stemmed_word = stemmer.stem(token)
        stemmed.append(token)
    # flatten list in stemmed))
    processed_sentence = ' '.join(stem for stem in stemmed)
    return processed_sentence


def extract_preferences(utterance):
    tokenized = word_tokenize(utterance)  # sentence broken into words
    no_stopwords = [word for word in tokenized if not word in stopwords.words()]  # remove stopwords for this part
    # print(no_stopwords)
    food, area, pricerange = "", "", ""

    for word in no_stopwords:
        for foods in food_list:
            if nltk.edit_distance(word, foods) <= 2:
                food = word
        for areas in area_list:
            if nltk.edit_distance(word, areas) <= 2:
                area = word
        for prices in pricerange_list:
            if nltk.edit_distance(word, prices) <= 2:
                pricerange = word

    preferences = [food, area, pricerange]
    print("Preferences", preferences)

    # dontcare case?

    return preferences


def lookup_restaurants(state):
    """Looks up restaurants from the restaurant_info.csv, based on the state
    	state: dictionary containing the preferences, is of type dict()
    	Returns: one restaurant and alternatives, both of type pd.DataFrame"""
    # Load database
    res_df = pd.read_csv("restaurant_info.csv")

    # If no preference is expressed, any pricerange will do
    if state["pricerange"] == "dontcare":
        price_cond = True
    else:
        price_cond = (res_df["pricerange"] == state["pricerange"]
                      )
    if state["area"] == "dontcare":
        area_cond = True
    else:
        area_cond = (res_df["area"] == state["area"])

    if state["food"] == "dontcare":
        food_cond = True
    else:
        food_cond = (res_df["food"] == state["food"])

    all_restaurants = res_df[price_cond & area_cond & food_cond]
    # If none are found, return an empty dataframe
    if all_restaurants.empty:
        return all_restaurants, all_restaurants

    # Randomly sample one from the restaurants
    restaurant = all_restaurants.sample(1)

    # Alternatives are all found restaurants excluding the sampled one
    alternatives = res_df.iloc[all_restaurants.index.difference(restaurant.index)]

    return restaurant, alternatives


def dialog_management(state, utterance, preferences):
    processed_utterance = preprocess(utterance)
    reply = ""
    file = open('B:/Documents/Master AI/INFOMAIR/venv/src/saved models/decision/decision_model.pkl', 'rb')
    model = pickle.load(file)

    vectorizer = CountVectorizer()
    bow_wrds = vectorizer.fit_transform([processed_utterance]).toarray()
    bow_wrds = pad_sequences(bow_wrds, maxlen=767, value=0)
    utterance_class = dictionary[np.argmax(model.predict(bow_wrds))]
    print("Utterance class", utterance_class)
    print("State", state)

    if state == 2:  # Zeusbot finished greeting the guest and we get their first reply
        utterance_class = "inform"
        if utterance_class == "inform":
            preference_list = extract_preferences(utterance)
            reply = "Hi, so you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                    " food in the " + preference_list[1] + " part? \n "
            next_state = 6

        return next_state, reply, preference_list
    # generate and assign reply

    if state == 6:  # Zeus bot asked if they got their preferences right and we get their reply
        utterance_class = "ack"
        if utterance_class == "deny" or utterance_class == "negate":
            preference_list = ["", "", ""]
            reply = "I am so sorry we could not help you, I am going to reboot my menory and let's try again :) \n " \
                    "maybe it helps if you slightly rephrase your sentences because I am silly."
        if utterance_class == "affirm" or utterance_class == "ack":
            if preferences[0] == "":
                reply = "You have not specified a preferred type of food, would you like to?"
                next_state = 25
            elif preferences[1] == "":
                reply = "You have not specified a preferred location, would you like to?"
                next_state = 25
            elif preferences[2] == "":
                reply = "You have not specified a preferred price range, would you like to?"
                next_state = 25
            else:
                preferences_dict = {"food": preferences[0], "area": preferences[1], "pricerange": preferences[2]}
                print("Dict", preferences_dict)
                restaurant, alternatives = lookup_restaurants(preferences_dict)
                print(restaurant, alternatives)
                reply = "I think you would really like" + restaurant + "is this agreeable?"
                next_state = 12

    if state == 12:  # Zeusbot suggested a restaurant and we get their reply
        if utterance_class == "affirm":
            reply = "Thank you for choosing Zeus Bot, I hope you enjoy your dinner. Goodbye."
            next_state = 17
        elif utterance_class == "reqalt":
            preferences_dict = {"food": food, "area": area, "pricerange": pricerange}
            restaurant, alternatives = lookup_restaurants(preferences_dict)
            reply = "I think you would really like" + alternatives + "is this agreeable? If you keep finding me suggesting the same restaurant, maybe try again and ask for something different."
            next_state = 12
        elif utterance_class == "deny":
            reply = "I'm sorry I couldn't help you this time, let's start over! :) \n Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?"
            next_state = 1
            preferences = ["", "", ""]

        else:
            reply = "I'm sorry but I don't understand, let's start over \n Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?"
            next_state = 2
            preferences = ["", "", ""]
                
    print("return ")
    return next_state, reply, preferences
  

if __name__ == "__main__":
    df = pd.read_csv("B:/Documents/Master AI/INFOMAIR/venv/src/restaurant_info.csv")
    df = df.fillna('')  # replace nan with empty string
    area_list = set(df['area'].tolist())
    pricerange_list = set(df['pricerange'].tolist())
    food_list = set(df['food'].tolist())
    preferences = ["","",""]
    # print("Area List", area_list)
    # print("Pricerange list", pricerange_list)
    # print("Food list", food_list)

    # model = load_model("B:/Documents/Master AI/INFOMAIR/venv/src/saved models/feedforward")

    state = 2
    print("Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?")
    while True:

        user_input = input().lower()
        if user_input == "quit":
            break
        state, reply, preferences = dialog_management(state, user_input, preferences)
        print(reply)

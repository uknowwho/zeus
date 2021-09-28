import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# nltk.download('punkt')
# nltk.download('stopwords')

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
  14: "thankyou"
}


def preprocess(sentence):
    """Preprocesses sentence: normalizes whitespace, removes punctionation and stems words (removing inflection).
        sentence:	string, as input by user
        returns:	sentence as string, preprocessed"""
    # preprocessing
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # remove punctuation
    stemmer = PorterStemmer()
    tokenized = tokenizer.tokenize(sentence)

    stemmed = []
    for token in tokenized:
        stemmed_word = stemmer.stem(token)
        stemmed.append(stemmed_word)
    # flatten list in stemmed))
    processed_sentence = ' '.join(stem for stem in stemmed)
    return processed_sentence


def generate_reply(df):
    reply = "I think you would really like " + df["restaurantname"].to_string(index=False) + ", its located at  " + \
            df["addr"].to_string(index=False) + ", " + df["postcode"].to_string(index=False) + \
            " in the " + df["area"].to_string(index=False) + " and the phone number is " + \
            df["phone"].to_string(index=False) + ". \n  Do you agree? If you keep finding me " \
                                    "suggesting the same restaurant, maybe try again and ask for something different."

    return reply


def extract_preferences(utterance):
    tokenized = word_tokenize(utterance)  # sentence broken into words
    no_stopwords = [word for word in tokenized if not word in stopwords.words()]  # remove stopwords for this part
    food, area, pricerange = "", "", ""

    # Plain lookup
    for word in no_stopwords:
        if word in food_list:
            food = word

        elif word in area_list:
            area = word

        elif word in pricerange_list:
            pricerange = word

    if food == "":
        # Rules for food
        try:
            # matches strings like "spanish food" or "spanish restaurant"
            word = re.search(r"(\w+)(\sfood|\srestaurant)", utterance).group(1)

            if word == "any":
                food = "dontcare"

            else:
                # randomly choose a food that is closest to the matched word
                one_dist = []
                two_dist = []
                for foods in food_list:
                    distance = nltk.edit_distance(word, foods)
                    if distance == 2:
                        two_dist.append(foods)
                    if distance == 1:
                        one_dist.append(foods)

                if one_dist != []:
                    food = np.random.choice(one_dist)

                elif two_dist != []:
                    food = np.random.choice(two_dist)

        except AttributeError:
            food = ""

    # correct spelling mistakes
    if area == "":
        try:
            # matches strings like "spanish food" or "spanish restaurant"
            word = re.search(r"(\w+)(\sarea)", utterance).group(1)

            # if no preference
            if word == "any":
                area = "dontcare"

            else:
                for word in no_stopwords:
                    for areas in area_list:
                        if nltk.edit_distance(word, areas) <= 2:
                            area = areas

        except AttributeError:
            area = ""

    # correct spelling mistakes and detect -ly variants
    if pricerange == "":
        try:
            word = re.search(r"(\w+)(\spricerange)", utterance).group(1)

            if word == "any":
                pricerange = "dontcare"

            else:
                for word in no_stopwords:
                    for prices in pricerange_list:
                        if nltk.edit_distance(word, prices) <= 2:
                            pricerange = prices

        except AttributeError:
            pricerange = ""

    preferences = [food, area, pricerange]

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
    next_state = 0
    preference_list = preferences

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
            reply = "So you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                    " food in the " + preference_list[1] + " part? \n "
            next_state = 6
    # generate and assign reply

    elif state == 6:  # Zeus bot asked if they got their preferences right and we get their reply
        utterance_class = "ack"
        if utterance_class == "deny" or utterance_class == "negate":
            preference_list = ["", "", ""]
            reply = "I am so sorry we could not help you, I am going to reboot my menory and let's try again :) \n " \
                    "maybe it helps if you slightly rephrase your sentences because I am silly."
            exit()
        if utterance_class == "affirm" or utterance_class == "ack":
            preference_list = [preferences[0], preferences[1], preferences[2]]
            if preferences[0] == "":
                reply = "You have not specified a preferred type of food, you can enter a type or dontcare"
                next_state = 23
            elif preferences[1] == "":
                reply = "You have not specified a preferred location,  you can enter a location or type dontcare"
                next_state = 24
            elif preferences[2] == "":
                reply = "You have not specified a preferred price range,  you can enter a price or type dontcare"
                next_state = 25
            else:
                preferences_dict = {"food": preferences[0], "area": preferences[1], "pricerange": preferences[2]}
                # print("Dict", preferences_dict)
                restaurant, alternatives = lookup_restaurants(preferences_dict)
                # print(restaurant, alternatives)

                reply = generate_reply(restaurant)

                next_state = 12

    # user still needs to specify some food  preference
    elif state == 23:
        if utterance == "dontcare" or utterance == "dont care":
            food_pref = "dontcare"
        else:
            food_pref = extract_preferences(utterance)[0]
        preference_list = [food_pref, preferences[1], preferences[2]]
        reply = "Hi, so you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " part? \n "
        next_state = 6

    # user still needs to specify some area preference
    elif state == 24:
        if utterance == "dontcare" or utterance == "dont care":
            area_pref = "dontcare"
        else:
            area_pref = extract_preferences(utterance)[1]
        preference_list = [preferences[0], area_pref, preferences[2]]
        print(preference_list)
        reply = "Hi, so you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " part? \n "
        next_state = 6

    # user still needs to specify some pricerange preference
    elif state == 25:
        if utterance == "dontcare" or utterance == "dont care":
            pricerange_pref = "dontcare"
        else:
            pricerange_pref = extract_preferences(utterance)[2]
        preference_list = [preferences[0], preferences[1], pricerange_pref]
        reply = "Hi, so you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " part? \n "
        next_state = 6

    elif state == 12:  # Zeusbot suggested a restaurant and we get their reply
        if utterance_class == "affirm" or utterance_class == "ack":
            print("Thank you for choosing Zeus Bot, I hope you enjoy your dinner. Goodbye.")
            next_state = 17
            exit()
        elif utterance_class == "reqalt":
            preferences_dict = {"food": food, "area": area, "pricerange": pricerange}
            restaurant, alternatives = lookup_restaurants(preferences_dict)

            if alternatives.empty:
                reply = "I am sorry but there are no alternatives. We can restart this chat, so please try something " \
                        "else"
                next_state = 1

            else:
                reply = generate_reply(alternatives)
                next_state = 12

        elif utterance_class == "deny" or utterance_class == "negate":
            reply = "I'm sorry I couldn't help you this time, let's start over! :) \n Welcome to Zeus bot, " \
                    "let me help you suggest a restaurant, do you have any preferences?"
            next_state = 1
            preference_list = ["", "", ""]

        else:
            reply = "I'm sorry but I don't understand, let's start over \n Welcome to Zeus bot, let me help you " \
                    "suggest a restaurant, do you have any preferences?"
            next_state = 2
            preference_list = ["", "", ""]

    return next_state, reply, preferences


if __name__ == "__main__":
    df = pd.read_csv("B:/Documents/Master AI/INFOMAIR/venv/src/restaurant_info.csv")
    df = df.drop_duplicates()
    df = df.fillna("")

    area_list = set(df['area'].tolist())
    area_list = [area for area in area_list if str(area) != 'nan']

    pricerange_list = set(df['pricerange'].tolist())
    food_list = set(df['food'].tolist())
    preferences = ["", "", ""]

    # model = load_model("B:/Documents/Master AI/INFOMAIR/venv/src/saved models/feedforward")

    state = 2
    print("Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?")
    while True:

        user_input = input().lower()
        if user_input == "quit":
            break
        state, reply, preferences = dialog_management(state, user_input, preferences)
        print(reply)

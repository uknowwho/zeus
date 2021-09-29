import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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


def ask_preferences(preferences):
    reply = "We currently know that you want: \n food type: " + preferences[0] + "\n area: " + preferences[1] +\
            " \n price range: " + preferences[2] + "\n Yes or no?"

    return reply


def generate_reply(df):
    # Bot couldnt find a restaurant so it starts over
    if df.values.size == 0:
        reply = "I am sorry, I could not find a restaurant to match your preferences, please try again"
        next_state = 2
    else:
        reply = "I think you would really like " + df["restaurantname"].to_string(index=False) + ", its located at  " + \
                df["addr"].to_string(index=False) + ", " + df["postcode"].to_string(index=False) + \
                " in the " + df["area"].to_string(index=False) + " and the phone number is " + \
                df["phone"].to_string(index=False) + ". \n  Do you agree? If you keep finding me " \
                                    "suggesting the same restaurant, maybe try again and ask for something different."
        next_state = 12
    return reply, next_state


def parse_match(match, property_list):
    """Return the closest matching preference word for a given food/area/pricerange word.
    Tries to find the closest match and chooses randomly in case of a tie.
	word: word that was matched as food/area/pricerange
	propertylist: list of possible properties, e.g. [north, south, ..] for area"""
    word = match.group(1)

    if word == "any":
        return "dontcare"

    one_dist = []
    two_dist = []
    for prop in property_list:
        distance = nltk.edit_distance(word, prop)
        if distance == 0:
            return word

        if distance == 1:
            one_dist.append(prop)

        if distance == 2:
            two_dist.append(prop)

    if one_dist != []:
        return np.random.choice(one_dist)

    elif two_dist != []:
        return np.random.choice(two_dist)

    return ""

def extract_preferences(utterance, preferences):
    """Takes in an utterance string that is expected to contain preferences for restaurant
    food type/area/pricerange, and returns all the preferences it can find.
    If a certain preference is not found, this is represented by the empty string.
	  utterance:	string of utterance with preferences expected
	  returns:	list, in the form [food, area, pricerange]"""
  
    food, area, pricerange = preferences[0], preferences[1], preferences[2]
    tokenized = word_tokenize(utterance)  # sentence broken into words
    # no_stopwords = tokenized
    no_stopwords = [word for word in tokenized if not word in stopwords.words()]  # remove stopwords for this part

    # Plain lookup
    for word in no_stopwords:
        if word in food_list:
            food = word

        elif word in area_list:
            area = word

        elif word in pricerange_list:
            pricerange = word


    # If the food was not in the list
    if not food:
        match = re.search(r"(\w+)(\sfood|\srestaurant)", utterance)

        if match:
            food = parse_match(match, food_list)


    # If the area was not in the list
    if not area:
        match = re.search(r"(\w+(?<!(ern)))((ern)*\sarea)", utterance)

        if match:
            area = parse_match(match, area_list)

    # If the
    if not pricerange:
        match = re.search(r"(\w+)(ly\spriced|\spricerange)", utterance)

        if match:
            pricerange = parse_match(match, pricerange_list)

    return [food, area, pricerange]


def lookup_restaurants(state):
    """Looks up restaurants from the restaurant_info.csv, based on the state
    state: dictionary containing the preferences, is of type dict()
    Returns: one restaurant and alternatives, both of type pd.DataFrame"""
    # Load database

    res_df = pd.read_csv('restaurant_info.csv')

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

    # load the model and the vectorier
    model = load_model('saved models/feedforward')
    file = open('saved models/vectorizer/vectorizer.pkl', 'rb')
    vectorizer = pickle.load(file)
    file.close()

    bow_wrds = vectorizer.transform([processed_utterance]).toarray()
    bow_wrds = pad_sequences(bow_wrds, maxlen=704, value=0)
    utterance_class = dictionary[np.argmax(model.predict(bow_wrds))]
    print("Utterance class", utterance_class)
    print("State", state)

    if utterance_class == "restart":
        reply = "I'm sorry I couldn't help you this time, let's start over! :) \n Welcome to Zeus bot, " \
                "let me help you suggest a restaurant, do you have any preferences?"
        next_state = 2
        preference_list = ["", "", ""]
        return next_state, reply, preference_list

    if state == 2:  # Zeusbot finished greeting the guest and we get their first reply

        if utterance_class == "inform":
            preference_list = extract_preferences(utterance, preference_list)
            print(preference_list)

            reply = ask_preferences(preference_list)
            next_state = 6
    # generate and assign reply

    elif state == 6:  # Zeus bot asked if they got their preferences right and we get their reply

        if utterance_class == "deny" or utterance_class == "negate":
            previous_preferences = preference_list
            preference_list = extract_preferences(utterance, preference_list)
            if preference_list == previous_preferences:
                reply = "I am so sorry we could not help you, I am going to reboot my menory and let's try again :) \n " \
                        "maybe it helps if you slightly rephrase your sentences because I am silly. \n Welcome to Zeus bot " \
                        "let me help you suggest a restaurant, do you have any preferences?"
                next_state = 2
            else:
                print(preference_list)
                print("Are these correct?")
                next_state = 6

        elif utterance_class == "affirm" or utterance_class == "ack":
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
                restaurant, alternatives = lookup_restaurants(preferences_dict)
                # print(restaurant, alternatives)

                reply, next_state = generate_reply(restaurant)
        elif utterance_class == "inform":
            preference_list = extract_preferences(utterance, preference_list)
            # if preference_list.count("") < 3:
            print(preference_list)
            print("Are these correct?")
            next_state = 6

        else:
            print("eoiruoibfo")

    # user still needs to specify some food  preference
    elif state == 23:
        if utterance == "dontcare" or utterance == "dont care":
            food_pref = "dontcare"
        else:
            food_pref = extract_preferences(utterance, preference_list)[0]
        preference_list = [food_pref, preferences[1], preferences[2]]
        reply = "So you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " ? \n "
        next_state = 6

    # user still needs to specify some area preference
    elif state == 24:
        print("start of state 24i w", preference_list)
        if utterance == "dontcare" or utterance == "dont care":
            area_pref = "dontcare"
            preference_list[1] = area_pref
        else:
            preference_list = extract_preferences(utterance, preference_list)
        print("after update: ",preference_list)
        reply = "So you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " ? \n "
        next_state = 6

    # user still needs to specify some pricerange preference
    elif state == 25:
        if utterance == "dontcare" or utterance == "dont care":
            pricerange_pref = "dontcare"
        else:
            pricerange_pref = extract_preferences(utterance, preference_list)[2]
        preference_list = [preferences[0], preferences[1], pricerange_pref]
        reply = "So you want a " + preference_list[2] + " restaurant that offers " + preference_list[0] + \
                " food in the " + preference_list[1] + " ? \n "
        next_state = 6

    elif state == 12:  # Zeusbot suggested a restaurant and we get their reply
        if utterance_class == "affirm" or utterance_class == "ack":
            print("Thank you for choosing Zeus Bot, I hope you enjoy your dinner. Goodbye.")
            next_state = 17
            exit()
        elif utterance_class == "reqalt" or utterance_class == "negate" or utterance_class == "deny":
            preferences_dict = {"food": preference_list[0], "area": preference_list[1], "pricerange": preference_list[2]}
            restaurant, alternatives = lookup_restaurants(preferences_dict)
            #STill need to finish this

            # if there are no alternative suggestions, restart
            if len(alternatives.index) == 0:
                reply = "I am sorry but there are no alternatives. We can restart this chat, so please try something " \
                        "else"
                next_state = 2

            else:
                reply = generate_reply(alternatives)
                next_state = 12

        else:
            reply = "I'm sorry I couldn't help you this time, let's start over! :) \n Welcome to Zeus bot, " \
                    "let me help you suggest a restaurant, do you have any preferences?"
            next_state = 2
            preference_list = ["", "", ""]

    else:
        reply = "I'm sorry but I don't understand, let's start over \n Welcome to Zeus bot, let me help you " \
                "suggest a restaurant, do you have any preferences?"
        next_state = 2
        preference_list = ["", "", ""]

    return next_state, reply, preference_list


if __name__ == "__main__":

    df = pd.read_csv('restaurant_info.csv')
    df = df.drop_duplicates()
    df = df.fillna('')

    food_list = set(df['food'].tolist())
    area_list = set(df['area'].tolist())
    area_list = [area for area in area_list if str(area) != 'nan']
    pricerange_list = set(df['pricerange'].tolist())

    preferences = ["", "", ""]

    model = load_model("saved models/feedforward")

    state = 2
    print("Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?")
    while True:

        user_input = input().lower()
        if user_input == "quit":
            break
        state, reply, preferences = dialog_management(state, user_input, preferences)
        print(reply)

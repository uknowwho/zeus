import re
import sys
import pandas as pd
import pyttsx3
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

WELCOME = "Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?"
SORRY = "I'm sorry I couldn't help you this time, let's start over! :) \n"

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


def debugprint(*args):
    """Print the arguments only if the global DEBUG option is set"""
    if "--debug" in sys.argv:
        print("DEBUG:", ' '.join((str(arg) for arg in args)))


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
    # flatten list in stemmed
    processed_sentence = ' '.join(stem for stem in stemmed)
    return processed_sentence


def majority_label(sent):
    res_df = pd.read_csv('updated_restaurant_info.csv')
    return res_df["labels"].value_counts().index[0]


def format_reply(df):

    reply = f"""I think you would really like {df['restaurantname'].to_string(index=False)},
it's located at {df['addr'].to_string(index=False)} {df['postcode'].to_string(index=False)}
in the  {df['area'].to_string(index=False)}  and the phone number is
{df['phone'].to_string(index=False)} Do you agree? If you find that I keep
suggesting the same restaurant, you could try again and ask for something different."""
    return reply


def generate_reply(df):
    """Generates a reply based on a specific restaurant.
        df:		information about a specific restaurant, stored in a dataframe
        returns:	recommendation or failure reply, string"""

    # Bot couldnt find a restaurant so it starts over
    if df.values.size == 0:
        reply = SORRY + WELCOME
        next_state = 2
    # A valid restaurant has been passed in, so it is recommended to the user
    else:
        reply = format_reply(df)
        next_state = 12

    return reply, next_state


def generate_reply_alternatives(df):
    if df.values.size == 0:
        reply = "I am sorry, I could not find an alternative restaurant to match your preferences, please try again"
        next_state = 2
    else:
        df = df.sample()
        reply = format_reply(df)
        next_state = 12

    return reply, next_state


def overlap(keywords, sentence):
    """Check whether any of the keywords are in the sentence"""
    return len(keywords.intersection(set(sentence.split()))) >= 1


def rule_based(sent):
    split_sent = sent.lower().split()

    if overlap({"looking", "searching", "want", "cheap", "north", "east",
                "south", "west", "priced", "find", "any", "italian", "moderate",
                "spanish", "matter"}, sent):
        return "inform"

    elif overlap({"what", "where", "when", "whats", "may", "could", "address",
                  "type", "phone", "code"}, sent):
        return "request"

    elif overlap({"thanks", "thank"}, sent):
        return "thankyou"

    elif overlap({"another", "other", "else"}, sent) or sent.startswith("how about"):
        return "reqalt"

    elif overlap({"um", "hm", "unintelligible", "noise"}, sent):
        return "null"

    elif overlap({"yes", "right", "yeah", "correct", "ye"}, sent):
        return "affirm"

    elif "bye" in sent:
        return "bye"

    elif sent.startswith("is it"):
        return "confirm"

    elif "hello" in sent or "hi" in split_sent:
        return "hello"

    elif "no" in split_sent:
        return "negate"

    elif "dont" in split_sent:
        return "deny"

    elif "repeat" in sent:
        return "repeat"

    elif "okay" in split_sent:
        return "ack"

    elif overlap({"restart", "start"}, sent):
        return "restart"

    elif "more" in split_sent:
        return "reqmore"

    return "inform"


def parse_match(match, property_list):
    """Finds the closest matching preference word for a given food/area/pricerange word.
    Tries to find the closest match and chooses randomly in case of a tie.
        word: 		word that was matched as food/area/pricerange
        propertylist:	list of possible properties, e.g. [north, south, ..] for area
        returns:		closest matching preference word as string"""

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

    if two_dist != []:
        return np.random.choice(two_dist)

    return ""


def extract_preferences(utterance, preferences):
    """Takes in an utterance string that is expected to contain preferences for restaurant
    food type/area/pricerange, and returns all the preferences it can find.
    If a certain preference is not found, this is represented by the empty string.
        utterance:	string of utterance with preferences expected
        returns:	list, in the form [food, area, pricerange]"""

    food, area, price = preferences[0], preferences[1], preferences[2]
    tokenized = word_tokenize(utterance)  # sentence broken into words
    no_stopwords = [word for word in tokenized if not word in stopwords.words()]  # remove stopwords for this part

    # Plain lookup
    for word in no_stopwords:
        if word in food_list:
            food = word

        elif word in area_list:
            area = word

        elif word in pricerange_list:
            price = word

    # If the food was not in the list look for patterns
    if not food and re.search(r"(\w+)(\sfood|\srestaurant)", utterance):
        food = parse_match(
            re.search(r"(\w+)(\sfood|\srestaurant)", utterance),
            food_list
        )

    # If the area was not in the list look for patterns
    if not area and re.search(r"(\w+(?<!(ern)))((ern)*\sarea)", utterance):
        area = parse_match(
            re.search(r"(\w+(?<!(ern)))((ern)*\sarea)", utterance),
            area_list
        )

    # If the pricerange was not in the list look for patterns
    if not price and re.search(r"(\w+)(ly\spriced|\spricerange)", utterance):
        price = parse_match(
            re.search(r"(\w+)(ly\spriced|\spricerange)", utterance),
            pricerange_list
        )

    return [food, area, price]


def get_bonus_preferences(utterance, bonus_preferences):
    """Retrieve bonus preferences from user utterance, or update if some are already known.
    Bonus preferences are, in order: good food, busy, long stay, romantic, children
        utterance:          user utterance, string
        bonus_preferences   list containing already known bonus preferences in order,
                            empty string if preference is not known
        returns:            list of the extra preferences the user gave
    """

    good_food, busy, longstay, romantic, children = bonus_preferences

    if "good" in utterance and "food" in utterance:
        good_food = True
    if "good" in utterance and "food" in utterance and "not" in utterance:
        good_food = False

    if "busy" in utterance and not "not" in utterance:
        busy = True
    if "busy" in utterance and "not" in utterance:
        busy = False

    if "long stay" in utterance and not "not" in utterance:
        longstay = True
    if "long stay" in utterance and "not" in utterance:
        longstay = False

    if "romantic" in utterance and not "not" in utterance:
        romantic = True
    if "romantic" in utterance and "not" in utterance:
        romantic = False

    if "children" in utterance and not "not" in utterance:
        children = True
    if "children" in utterance and "not" in utterance:
        children = False

    # Return the updated bonus preferences.
    return [good_food, busy, longstay, romantic, children]


def lookup_restaurants_bonus(restaurant, alternatives, bonus_preferences):
    """Looks up restaurants based on bonus preferences.
    Bonus preferences are: good food, busy, long stay, romantic, children
        restaurant:		    dataframe with the first suggestion
        alternatives:		dataframe with the other alternatives if they exists
        bonus_preferences: 	list with the extra preferences the user gave
        returns:		    a dataframe (could be empty) that satisfies the preferences"""

    all_restaurants = pd.concat([restaurant, alternatives])

    # good food
    if bonus_preferences[0]:
        all_restaurants = all_restaurants[all_restaurants['good food'] == True]
    elif bonus_preferences[0] == "":
        pass
    else:
        all_restaurants = all_restaurants[all_restaurants['good food'] == False]

    # busy
    if bonus_preferences[1]:
        all_restaurants = all_restaurants[all_restaurants.busy == True]
    elif bonus_preferences[1] == "":
        pass
    else:
        all_restaurants = all_restaurants[all_restaurants.busy == False]

    # long stay
    if bonus_preferences[2]:
        all_restaurants = all_restaurants[all_restaurants['long stay'] == True]
    elif bonus_preferences[2] == "":
        pass
    else:
        all_restaurants = all_restaurants[all_restaurants['long stay'] == False]

    # romantic

    if bonus_preferences[3]:
        all_restaurants = all_restaurants[all_restaurants.romantic == True]
    elif bonus_preferences[3] == "":
        pass
    else:
        all_restaurants = all_restaurants[all_restaurants.romantic == False]

    # children
    if bonus_preferences[4]:
        all_restaurants = all_restaurants[all_restaurants.children == True]
    elif bonus_preferences[4] == "":
        pass
    else:
        all_restaurants = all_restaurants[all_restaurants.children == False]

    # Randomly sample one from the restaurants
    if all_restaurants.values.size == 0:
        restaurant = all_restaurants
    else:
        restaurant = all_restaurants.sample(1)

    return restaurant


def lookup_restaurants(state):
    """Looks up restaurants from the updated_restaurant_info.csv, based on the state
        state: dictionary containing the preferences, is of type dict()
        returns: one restaurant and alternatives, both of type pd.DataFrame"""

    # Load database
    res_df = pd.read_csv('updated_restaurant_info.csv')

    # If no preference is expressed, any pricerange will do
    conds = {"food": True, "area": True, "pricerange": True}
    for prop in state:
        if state[prop] != "dontcare":
            conds[prop] = (res_df[prop] == state[prop])

    all_restaurants = res_df[conds["pricerange"] & conds["area"] & conds["food"]]

    # If none are found, return an empty dataframe
    if all_restaurants.empty:
        return all_restaurants, all_restaurants

    # Randomly sample one from the restaurants
    restaurant = all_restaurants.sample(1)

    # Alternatives are all found restaurants excluding the sampled one
    alternatives = res_df.iloc[all_restaurants.index.difference(restaurant.index)]
    # print(restaurant, alternatives)
    return restaurant, alternatives


def dontcare_check(utterance):
    """"Checks whether the utterance is "I don't care" or something close to that"""
    return nltk.edit_distance(utterance, "don't care") < 7


def confirm_preferences(preferences):
    """Produces a confirmation message
        preferences:    list of preferences in shape [foodtype, area, pricerange],
                        or equivalent dict
        returns:        confirmation message that asks whether all known preferences are correct"""

    # convert to dict if it isn't already, it's just neater
    if type(preferences) != dict:
        preferences = dict(zip(["food", "area", "pricerange"], preferences))

    # initiate "empty" message
    message = "So you want a"

    # add pricerange to message if known
    if preferences["pricerange"] != "" and preferences["pricerange"] != "dontcare":
        message = f"""{message} {preferences["pricerange"]}"""
    # dontcare is dealt with later

    message = f"{message} restaurant"

    # add food type if known
    if preferences["food"] == "dontcare":
        message = f"""{message} that offers any kind of food"""
    elif preferences["food"] != "":
        message = f"""{message} that offers {preferences["food"]} food"""

    # add area if known
    if preferences["area"] == "dontcare":
        message = f"""{message} in any area"""
    elif preferences["area"] != "":
        message = f"""{message} in the {preferences["area"]}"""

    if preferences["pricerange"] == "dontcare":
        message = f"""{message} for any price"""

    message = message + "?\n"

    return message


def dialog_management(state, utterance, preferences, bonus_preferences, alternatives, baseline=False):
    """Handles most of the dialog, the most important function that is repeatedly called
    in the main program.
        state:			    state number as integer, see the diagram
        utterance:		    user input as string
        preferences:		preferences known as of now, list
        bonus_preferences:	bonus preferences known as of now, list"""

    processed_utterance = preprocess(utterance)
    reply = ""
    next_state = 0
    # TODO: why do we do this renaming? it's confusing
    preference_list = preferences

    if not baseline:
        # load the model and the vectorier
        file = open('saved models/regression/logistic_regression.pkl', 'rb')
        model = pickle.load(file)
        file.close()
        file = open('saved models/vectorizer/vectorizer.pkl', 'rb')
        vectorizer = pickle.load(file)
        file.close()

        bow_wrds = vectorizer.transform([processed_utterance]).toarray()
        bow_wrds = pad_sequences(bow_wrds, maxlen=704, value=0)
        utterance_class = dictionary[model.predict(bow_wrds)[0]]

    else:
        print(processed_utterance)
        utterance_class = rule_based(processed_utterance)

    debugprint("Utterance class", utterance_class)
    debugprint("State", state)

    if utterance_class == "restart":
        reply = SORRY + WELCOME
        next_state = 2
        preference_list = ["", "", ""]
        return next_state, reply, preference_list, bonus_preferences

    elif state == 2:  # Zeusbot finished greeting the guest and we get their first reply

        if utterance_class == "inform":
            preference_list = extract_preferences(utterance, preference_list)
            debugprint(preference_list)

            reply = confirm_preferences(preference_list)
            next_state = 6
        if utterance_class == "affirm":
            reply = confirm_preferences(preference_list)
            next_state = 6
    # generate and assign reply

    # Zeus bot asked if they got their preferences right and we get their reply
    elif state == 6:

        if utterance_class == "deny" or utterance_class == "negate":
            previous_preferences = preference_list
            preference_list = extract_preferences(utterance, preference_list)
            if preference_list == previous_preferences:
                reply = SORRY + WELCOME
                next_state = 2
            else:

                print("Are these correct?")
                next_state = 6

        elif utterance_class == "affirm" or utterance_class == "ack":
            preference_list = [preference_list[0], preference_list[1], preference_list[2]]

            if preference_list[0] == "":
                reply = '''You have not specified a preferred type of food, you can enter a food type or say that you don't care.'''
                next_state = 23
            elif preference_list[1] == "":
                reply = '''You have not specified a preferred location,  you can enter a location or say that you don't care.'''
                next_state = 24
            elif preference_list[2] == "":
                reply = '''You have not specified a preferred price range,  you can enter a price or say that you don't care.'''
                next_state = 25
            else:
                next_state = 99
                reply = "Do you have any other preferences? Options are: \n" \
                        "   good food, busy, long stay, romantic, children"
        # user gave even more preferences
        elif utterance_class == "inform":
            preference_list = extract_preferences(utterance, preference_list)

            debugprint(preference_list)
            reply = "Are these correct?"
            next_state = 6
        else:
            print("Sorry, Zeus doesn't understand")
            debugprint(preference_list)
            reply = "Are these correct?"
            next_state = 6

    # user still needs to specify some food  preference
    elif state == 23:
        if dontcare_check(utterance):
            food_pref = "dontcare"
        else:
            food_pref = extract_preferences(utterance, preference_list)[0]
        preference_list = [food_pref, preferences[1], preferences[2]]
        reply = confirm_preferences(preference_list)
        next_state = 6

    # user still needs to specify some area preference
    elif state == 24:

        if dontcare_check(utterance):
            area_pref = "dontcare"
            preference_list[1] = area_pref
        else:
            preference_list = extract_preferences(utterance, preference_list)

        reply = confirm_preferences(preference_list)
        next_state = 6

    # user still needs to specify some pricerange preference
    elif state == 25:
        if dontcare_check(utterance):
            pricerange_pref = "dontcare"
        else:
            pricerange_pref = extract_preferences(utterance, preference_list)[2]
        preference_list = [preferences[0], preferences[1], pricerange_pref]

        reply = confirm_preferences(preference_list)
        next_state = 6

    # checking if user has any extra preferences
    elif state == 99:
        if utterance_class == "null":
            print("Im sorry Zeus doesn't understand \n")
            reply = "Do you have any extra preferences? Options are: \n good food, busy, long stay, romantic, children."
            next_state = 99

        elif utterance_class == "negate" or utterance_class == "deny":
            next_state = 12
            preferences_dict = dict(zip(["food", "area", "pricerange"], preferences))
            restaurant, alternatives = lookup_restaurants(preferences_dict)
            reply, next_state = generate_reply(restaurant)

        else:
            bonus_preferences = get_bonus_preferences(utterance, bonus_preferences)
            # print("bonus preferences after extratcion", bonus_preferences)
            preferences_dict = dict(zip(["food", "area", "pricerange"], preference_list))
            restaurant, alternatives = lookup_restaurants(preferences_dict)

            restaurant = lookup_restaurants_bonus(restaurant, alternatives, bonus_preferences)

            if restaurant.values.size == 0:
                reply = SORRY + WELCOME
                next_state = 2
                bonus_preferences = ["", "", "", "", ""]
            else:
                reply, next_state = generate_reply(restaurant)
                next_state = 12

    elif state == 12:  # Zeusbot suggested a restaurant and we get their reply
        if utterance_class == "affirm" or utterance_class == "ack":
            print("Thank you for choosing Zeus Bot, I hope you enjoy your dinner. Goodbye.")
            next_state = 17
            exit()
        elif utterance_class == "reqalt" or utterance_class == "negate" or utterance_class == "deny":

            reply, next_state = generate_reply_alternatives(alternatives)

        else:

            reply = SORRY + WELCOME
            next_state = 2
            preference_list = ["", "", ""]

    else:
        reply = SORRY + WELCOME
        next_state = 2
        preference_list = ["", "", ""]

    return next_state, reply, preference_list, bonus_preferences, alternatives


if __name__ == "__main__":
    t2s = False
    baseline = False
    if "--t2s" in sys.argv:
        t2s = True
        engine = pyttsx3.init()

    if "--baseline" in sys.argv:
        baseline = True

    df = pd.read_csv('updated_restaurant_info.csv')
    df = df.drop_duplicates()
    df = df.fillna('')

    area_list = set(df['area'].dropna().tolist()) | {'center'}
    pricerange_list = set(df['pricerange'].dropna().tolist())
    food_list = set(df['food'].dropna().tolist()) | {'world', 'swedish', 'danish'}

    preferences = ["", "", ""]
    bonus_preferences = ["", "", "", "", ""]
    alternatives = []

    file = open('saved models/regression/logistic_regression.pkl', 'rb')
    model = pickle.load(file)
    file.close()

    state = 2
    print("Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?")
    if t2s:
        engine.say("Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?")
        engine.runAndWait()

    while True:

        user_input = input().lower()
        if user_input == "quit":
            break

        state, reply, preferences, bonus_preferences, alternatives = dialog_management(state, user_input, preferences,
                                                                         bonus_preferences, alternatives, baseline)
        debugprint(state)
        print(reply)
        if t2s:
            engine.say(reply)
            engine.runAndWait()

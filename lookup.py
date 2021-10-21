import re
import pandas as pd
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


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
        if token == "expensive":
            stemmed_word = "expensive"
        else:
            stemmed_word = stemmer.stem(token)
        stemmed.append(stemmed_word)
    # flatten list in stemmed
    processed_sentence = ' '.join(stem for stem in stemmed)
    return processed_sentence

# Functions that generate replies


def overlap(keywords, sentence):
    """Check whether any of the keywords are in the sentence"""
    return len(keywords.intersection(set(sentence.split()))) >= 1


def parse_match(match, property_list):
    """Finds the closest matching preference word for a given food/area/pricerange word.
    Tries to find the closest match and chooses randomly in case of a tie.
        word: 		word that was matched as food/area/pricerange
        propertylist:	list of possible properties, e.g. [north, south, ..] for area
        returns:		closest matching preference word as string"""

    word = match.group(1)

    if word == "ani":
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


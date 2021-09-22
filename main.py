import pandas as pd


def generate_reply(state):
	return reply

def extract_preferences(utterance):
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
		price_cond = (res_df["pricerange"] == state["pricerange"])

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


def levenshtein(utterance):
	return preferences



def dialog_management(state, utterance):
	# pre-process utterance
	# classify utterance

	# if inform:
		# extract preferences using rules
		# assign next_state
		# if not found:
			# use levenshtein

		# generate and assign reply


	return next_state, reply

if __name__ == "__main__":
	state = []
	while True:
		user_input = input().lower()
		if user_input == "quit":
			break

		state, reply = dialog_management(state, user_input)

		print(reply)

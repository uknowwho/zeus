def generate_reply(state):
	return reply

def extract_preferences(utterance):
	return preferences

def lookup_restaurants(preferences):
	return restaurant

def levenshtein(utterance):
	return preferences


def preprocess(sentence):
  processed_sentence = []
  # preprocessing 
	# remove punctuation and stem the words into their roots
	# tokenizer breaks the sentences into words
  tokenizer = nltk.RegexpTokenizer(r"\w+") # remove punctuation
  stemmer = PorterStemmer()
  tokenized = tokenizer.tokenize(sentence)
  stemmed = []
  for token in tokenized:
    stemmed_word  = stemmer.stem(token)
    stemmed.append(stemmed_word)
  # flatten list
  processed_sentence.append(" ".join(stem for stem in stemmed))
  return processed_sentence


def dialog_management(state, preferences, utterance):
	# pre-process utterance
	processed_utterance = preprocess(utterance)
	
	reply = ""
	
	
	
	# classify utterance ToDo: Load model into main somewhere, feed processed_utterance to model, return class.
	utterance_class = model(processed_utterance)

	
	if state == 1:
		reply = "Welcome to Zeus bot, let me help you suggest a restaurant, do you have any preferences?"
		next_state = 2

	if state == 2: #Zeusbot finished greeting the guest and we get their first reply
		if utterance_class == "inform"
			preference_list = extract_preferences(utterance)
			if len(preference_list[0]) > 0:
				reply = "Hi so you want a restaurant that offers" + preference_list[0] + "food?"
				next_state = 6
			if len(preference_list[1]) > 0:
				reply += "So you want a restaurant in the" + preference_list[1] + "part of town?"
				next_state = 6
			if len(preference_list[2]) > 0:
				reply += "So you want a restaurant that is" + preference_list[2] + "price range?"
				next_state = 6
		# extract preferences using rules
		# assign next_state
		# if not found:
			# use levenshtein

		# generate and assign reply
	if state == 6: #Zeus bot asked if they got their preferences right and we get their reply
		if utterance_class == "deny" or utterance_class == "negate":
			preferences = [" ", " ", " "]
			reply = "I am so sorry we could not help you, I am going to reboot my menory and let's try again :) \n maybe it helps if you slightly rephrase your sentences because I am silly."
		if utterance_class == "affirm":
			if preferences[0] == " ":
				reply = "You have not specified a preferred type of food, would you like to?"
				next_state = 25
			elif preferences[1] == " ":
				reply = "You have not specified a preferred location, would you like to?"
				next_state = 25
			elif preferences[2] == " ":
				reply = "You have not specified a preferred price range, would you like to?"
				next_state = 25
			else :
				restaurant = lookup_restaurants(preferences)
				reply = "I think you would really like" + restaurant + "is this agreeable?"
				next_state = 12

	
	
	if state == 12: # Zeusbot suggested a restaurant and we get their reply
		if utterance_class == "affirm":
			reply = "Thank you for choosing Zeus Bot, I hope you enjoy your dinner. Goodbye."
			next_state = 17
		elif utterance_class == "reqalt":
			restaurant = lookup_restaurants(preferences)
			reply = "I think you would really like" + restaurant + "is this agreeable? If you keep finding me suggesting the same restaurant, maybe try again and ask for something different."
			next_state = 12
		elif utterance_class == "deny":
			reply = "I'm sorry I couldn't help you this time, let's start over! :)"
			next_state = 1
			preferences = [" "," "," "]
		
		else: 
			reply= "I'm sorry but I don't understand, please rephrase your question"


	return next_state, preferences, reply

if __name__ == "__main__":
	state = []
	while True:
		user_input = input().lower()
		if user_input == "quit":
			break

		state, reply = dialog_management(state, user_input)

		print(reply)

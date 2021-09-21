def generate_reply(state):
	return reply

def extract_preferences(utterance):
	return preferences

def lookup_restaurants(state):
	return restaurant

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

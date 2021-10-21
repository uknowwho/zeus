import sys
WELCOME = "Welcome to Zeus bot, let me help you suggest a restaurant, please begin by stating your preferences."
SORRY = "I'm sorry I couldn't help you this time, let's start over! :) \n"


def debugprint(*args):
    """Print the arguments only if the global DEBUG option is set"""
    if "--debug" in sys.argv:
        print("DEBUG:", ' '.join((str(arg) for arg in args)))



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

def acknowledge_bonusses(bonus_preferences):
    #generate a string that will let the user know we will look for their preferences
    # receive a list of booleans [good food, busy, long stay, romantic, children]
    
    reply = "I will now try to find a restaurant that "
    if bonus_preferences[0] == "":
      pass
    elif bonus_preferences[0]:
      bonus = "offers good food, "
      reply += bonus
    else:
      bonus = "offers bad food, "
      reply += bonus
  
    if bonus_preferences[1] == "":
      pass
    elif bonus_preferences[1]:
      bonus = "is busy, "
      reply += bonus
    else:
      bonus = "is not busy, "
      reply += bonus
  
    if bonus_preferences[2] == "":
      pass
    elif bonus_preferences[2]:
      bonus = "offers long stay,, "
      reply += bonus
    else:
      bonus = "is fast, "
      reply += bonus
  
    if bonus_preferences[3] == "":
      pass
    elif bonus_preferences[3]:
      bonus = "is romantic, "
      reply += bonus
    else:
      bonus = "is not romantic, "
      reply += bonus
  
  
  
    if bonus_preferences[4] == "":
      pass
    elif bonus_preferences[4]:
      bonus = "is child-friendly "
      reply += bonus
    else:
      bonus = "is not child-friendly, "
      reply += bonus



    return reply

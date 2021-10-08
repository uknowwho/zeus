import pandas as pd


def majority_label():
    res_df = pd.read_csv('updated_restaurant_info.csv')
    return res_df["labels"].value_counts().index[0]


def rule_based(sent):
    split_sent = sent.split()

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

    elif overlap({"yes", "right", "yeah"}, sent):
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
    else:
        return majority_label()


import pandas as pd
import numpy as np

res_df = pd.read_csv("restaurant_info.csv")
res_df["good food"] = np.random.choice([True, False], size=(res_df.shape[0]))
res_df["busy"] = np.random.choice([True, False], size=(res_df.shape[0]))
res_df["long stay"] = np.random.choice([True, False], size=(res_df.shape[0]))
res_df["romantic"] = np.random.choice([True, False], size=(res_df.shape[0]))
res_df["children"] = np.random.choice([True, False], size=(res_df.shape[0]))

res_df.loc[res_df["food"] == "spanish", "long stay"] = True
res_df.loc[(res_df["pricerange"] == "cheap") & (res_df["good food"] == True), "busy"] = True
res_df.loc[res_df["busy"] == True, "long stay"] = True
res_df.loc[res_df["long stay"] == True, "children"] = False
res_df.loc[res_df["long stay"] == True, "romantic"] = True
res_df.loc[res_df["busy"] == True, "romantic"] = False

res_df[res_df["long stay"] == True]
res_df.to_csv("updated_restaurant_info.csv", index=False, header=res_df.columns)
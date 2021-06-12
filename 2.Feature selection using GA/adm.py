from main import main
import pandas as pd
l=main()

df=pd.read_csv('labelled_data.csv')
l.append('diagnosis')
new_f = df[l]
new_f.to_csv("feature_selected.csv", index=False)

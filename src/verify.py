import pandas as pd
df = pd.read_csv("D:/Code wala scene/techsphere-analytics/data/refined/jobs_nlp.csv")
print(df.head())
print(df.info())
print(df['domain'].value_counts())
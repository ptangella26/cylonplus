import pandas as pd
import datetime

def format_date(x):
    month_date = {
        'January': (1, 31),
        'February': (2, 28), 
        'March': (3, 31),
        'April': (4, 30),
        'May': (5, 31),
        'June': (6, 30),
        'July': (7, 31),
        'August': (8, 31),
        'September': (9, 30),
        'October': (10, 31),
        'November': (11, 30),
        'December': (12, 31)
    }
    month, year = x['Period'].split()
    year = int(year)
    if year % 4 == 0 and month == "February":
        day = 29
    else:
        day = month_date[month][1]
    return pd.Timestamp(month=month_date[month][0], day=day, year=year)

df = pd.read_csv('us_carrier_2000_2024.csv')
df['ds'] = df.apply(format_date, axis=1)
df['unique_id'] = 1
df = df.drop(['Period'], axis=1)
df.rename(columns={'Total': 'y'}, inplace=True)
df.set_index('ds')
df.to_csv('us_carrier_passenger.csv', index=False)



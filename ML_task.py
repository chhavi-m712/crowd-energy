import numpy as np
import pandas as pd
test_dataset = pd.read_csv('C:/Users/chhavi mittal/Downloads/tour_logs_train.csv')
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
dataframe = pd.DataFrame(test_dataset)
def clean_data_set(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe[dataframe['Ticket_Price'] != 'Free']
    def extract_currency_amount(x):
        if pd.isna(x):
            return pd.Series([np.nan, np.nan])

        x = str(x).strip()

        # extract FIRST numeric value (handles VIP text, commas, etc.)
        match = re.search(r"\d+\.?\d*", x)
        if not match:
            return pd.Series([np.nan, np.nan])

        amount = float(match.group())

        # detect currency
        if "£" in x:
            currency = "GBP"
        elif "€" in x:
            currency = "EUR"
        elif "$" in x:
            currency = "USD"
        elif "USD" in x:
            currency = "USD"
        else:
            currency = "USD"   # assume USD if not specified

        return pd.Series([currency, amount])
    dataframe[["currency", "amount"]] = dataframe["Ticket_Price"].apply(
        extract_currency_amount
    )
    exchange_rates = {
        "USD": 1.00,
        "GBP": 1.27,
        "EUR": 1.09
    }
    dataframe["Ticket_Price_USD"] = dataframe["amount"] * dataframe["currency"].map(exchange_rates)
    dataframe.drop(columns=["Ticket_Price", "currency", "amount"], inplace=True)

    raw = dataframe['Show_DateTime'].astype(str).str.lower().str.strip()

    # 1️⃣ First flexible parse (handles most formats)
    parsed = pd.to_datetime(
        raw,
        errors='coerce',
        dayfirst=True
    )

    today = pd.Timestamp.now().normalize()

    # 2️⃣ Word-based times
    time_words = {
        'late night': 23,
        'night': 21,
        'evening': 18,
        'afternoon': 16,
        'morning': 10
    }

    # yesterday night / yesterday evening
    for word, hour in time_words.items():
        mask = parsed.isna() & raw.str.contains(f'yesterday.*{word}', na=False)
        parsed.loc[mask] = (today - pd.Timedelta(days=1)) + pd.Timedelta(hours=hour)

    # standalone words
    for word, hour in time_words.items():
        mask = parsed.isna() & raw.str.fullmatch(word)
        parsed.loc[mask] = today + pd.Timedelta(hours=hour)

    # 3️⃣ dd-mm-yyyy hh:mm AM/PM
    mask = parsed.isna() & raw.str.match(
        r'\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s*(am|pm)',
        na=False
    )
    parsed.loc[mask] = pd.to_datetime(
        raw[mask],
        format='%d-%m-%Y %I:%M %p',
        errors='coerce'
    )

    # 4️⃣ dd-mm-yyyy (date only)
    mask = parsed.isna() & raw.str.match(r'\d{1,2}-\d{1,2}-\d{4}$', na=False)
    parsed.loc[mask] = pd.to_datetime(
        raw[mask] + ' 18:00',
        format='%d-%m-%Y %H:%M',
        errors='coerce'
    )

    # 5️⃣ dd/mm/yyyy or mm/dd/yyyy
    mask = parsed.isna() & raw.str.match(r'\d{1,2}/\d{1,2}/\d{4}', na=False)
    parsed.loc[mask] = pd.to_datetime(
        raw[mask],
        errors='coerce'
    )

    # 6️⃣ Month name formats (june 4,2025 etc.)
    mask = parsed.isna() & raw.str.contains(r'[a-z]', regex=True, na=False)
    parsed.loc[mask] = pd.to_datetime(
        raw[mask],
        errors='coerce'
    )

    # 7️⃣ Final safety net (NO NaT allowed)
    parsed.fillna(today + pd.Timedelta(hours=18), inplace=True)

    # 8️⃣ Enforce final format ONLY ONCE
    dataframe['Show_DateTime'] = parsed.dt.strftime('%m-%d-%Y %H:%M')
    # Volume_Level
    dataframe['Volume_Level'] = dataframe.groupby('Venue_ID')['Volume_Level']\
        .transform(lambda x: x.fillna(x.mean()))

    dataframe['Volume_Level'] = dataframe['Volume_Level'].fillna(
        dataframe['Volume_Level'].mean()
    )

    # Crowd_Size
    dataframe['Crowd_Size'] = dataframe.groupby('Venue_ID')['Crowd_Size']\
        .transform(lambda x: x.fillna(x.mean()))

    dataframe['Crowd_Size'] = dataframe['Crowd_Size'].fillna(
        dataframe['Crowd_Size'].mean()
    )

    return dataframe

dataframe = clean_data_set(dataframe)
dataframe.dropna(subset=['Crowd_Energy'], inplace=True)
Q1 = dataframe['Crowd_Energy'].quantile(0.25)
Q3 = dataframe['Crowd_Energy'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

dataframe = dataframe[
    (dataframe['Crowd_Energy'] >= lower) &
    (dataframe['Crowd_Energy'] <= upper)
]

#print(print(dataframe.isna().sum().sort_values(ascending=False)))

'''
sns.histplot(dataframe['Crowd_Energy'], kde=True)
plt.title('Distribution of Crowd Energy')
plt.show()
'''

'''
sns.histplot(dataframe['Merch_Sales_Post_Show'], kde=True)
plt.title('Distribution of Merch Sales Post Show')
plt.show()

'''
'''
dataframe['hour'] = pd.to_datetime(dataframe['Show_DateTime']).dt.hour

sns.barplot(x='hour', y='Crowd_Energy', data=dataframe)
plt.title('Crowd Energy vs Show Hour')
plt.show()
'''

'''
sns.barplot(x='Venue_ID',y='Crowd_Energy',data=dataframe)
plt.title('Crowd Energy VS Venue')
plt.show()
'''
'''
sns.barplot(x='Day_of_Week',y='Crowd_Energy',data=dataframe)
plt.title('Crowd_Energy VS day')
plt.show()
'''

'''
dataframe['Moon_Phase_Short'] = dataframe['Moon_Phase'].map({
    'Waning Crescent': 'WC',
    'New Moon': 'NM',
    'Full Moon': 'FM',
    'First Quarter': 'FQ',
    'Last Quarter': 'LQ',
    'Waxing Gibbous': 'WG',
    'Waxing Crescent': 'WCr'
})

sns.barplot(x='Moon_Phase_Short',y='Crowd_Energy',data=dataframe)
plt.title('Crowd_Energy VS moon')
plt.show()
'''
'''
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=dataframe,
    x='Ticket_Price_USD',
    y='Crowd_Energy'
)

plt.title('Ticket Price vs Crowd Energy')
plt.xlabel('Ticket Price (USD)')
plt.ylabel('Crowd Energy')
plt.tight_layout()
plt.show()
'''
'''
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=dataframe,
    x='Ticket_Price_USD',
    y='Merch_Sales_Post_Show'
)

plt.title('Ticket Price vs Merch_Sales')
plt.xlabel('Ticket Price (USD)')
plt.ylabel('Merch Sales')
plt.tight_layout()
plt.show()
'''
'''
sns.barplot(x='Crowd_Size',y='Crowd_Energy',data=dataframe)
plt.title('Crowd_Energy VS Crowd_size')
plt.show()
'''
'''
sns.scatterplot(
    data=dataframe,
    x='Crowd_Size',
    y='Crowd_Energy'
)

plt.title('Crowd size vs crowd energy')
plt.xlabel('crowd size')
plt.ylabel('crowd energy')
plt.tight_layout()
plt.show()
'''
'''
sns.barplot(x='Opener_Rating',y='Crowd_Energy',data=dataframe)
plt.title('Crowd energy vs opener')
plt.show()
'''
'''
sns.scatterplot(
    data=dataframe,
    x='Merch_Sales_Post_Show',
    y='Crowd_Energy'
)

plt.title('Merch Sale vs crowd energy')
plt.xlabel('Merch Sale')
plt.ylabel('crowd energy')
plt.tight_layout()
plt.show()
'''
'''
sns.barplot(x='Band_Outfit', y='Crowd_Energy', data=dataframe)
plt.title('Crowd Energy vs Band outfit')
plt.show()
'''
'''
sns.barplot(x='Weather', y='Crowd_Energy', data=dataframe)
plt.title('Crowd Energy vs Weather')
plt.show()
'''
'''
venues = dataframe["Venue_ID"].unique()
n = len(venues)

fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 4*n), sharex=True)

# If there's only one venue, axes is not a list
if n == 1:
    axes = [axes]

for ax, venue in zip(axes, venues):
    group = dataframe[dataframe["Venue_ID"] == venue] \
                .sort_values("Ticket_Price_USD")
    
    ax.scatter(
        group["Ticket_Price_USD"],
        group["Crowd_Energy"],
        alpha=0.6
    )
    
    ax.plot(
        group["Ticket_Price_USD"],
        group["Crowd_Energy"],
        alpha=0.5
    )
    
    ax.set_title(f"Venue: {venue}")
    ax.set_ylabel("Crowd_Energy")
    ax.grid(True)

axes[-1].set_xlabel("Ticket_Price_USD")
plt.tight_layout()
plt.show()
'''
'''
#Training
X = dataframe.drop(columns=['Crowd_Energy','Merch_Sales_Post_Show'])
y = dataframe['Crowd_Energy']
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

lin_reg = LinearRegression()

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', lin_reg)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
model.fit(X_train, y_train)
joblib.dump(model, "crowd_energy_model.pkl")

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

train_df = pd.read_csv("tour_logs_test_input.csv")

train_df = clean_data_set(train_df)


model = joblib.load("crowd_energy_model.pkl")

train_df['Predicted_Crowd_Energy'] = model.predict(train_df)
train_df.to_csv("predicted_crowd_energy.csv", index=True)
prediction = train_df['Predicted_Crowd_Energy']
prediction.to_csv("predicted.csv", index=True)
'''
'''
X = dataframe[['Merch_Sales_Post_Show']]   # double brackets = DataFrame
y = dataframe['Crowd_Energy']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model.fit(X_train, y_train)
joblib.dump(model, "crowd_energy_model.pkl")

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
'''
# ----------------------------
# 1. Define X and y
# ----------------------------
X = dataframe.drop(columns=['Crowd_Energy','Merch_Sales_Post_Show'])
y = dataframe['Crowd_Energy']

# ----------------------------
# 2. Identify column types
# ----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# ----------------------------
# 3. Preprocessing
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),  # RF does NOT need scaling
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# ----------------------------
# 4. Random Forest model
# ----------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# ----------------------------
# 5. Pipeline
# ----------------------------
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', rf)
])

# ----------------------------
# 6. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 7. Train model
# ----------------------------
model.fit(X_train, y_train)

# ----------------------------
# 8. Save model
# ----------------------------
joblib.dump(model, "crowd_energy_random_forest.pkl")

# ----------------------------
# 9. Evaluate
# ----------------------------
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

train_df = pd.read_csv("tour_logs_test_input.csv")

train_df = clean_data_set(train_df)


model = joblib.load("crowd_energy_model.pkl")

train_df['Predicted_Crowd_Energy'] = model.predict(train_df)
train_df.to_csv("predicted_crowd_energy.csv", index=True)
prediction = train_df['Predicted_Crowd_Energy']
prediction.to_csv("predicted.csv", index=True)

































import pandas as pd

# Lookback time feature
def add_time_diff(df, ori_col, shift_count):
    shift = df[ori_col].shift(shift_count)
    new_col = ori_col + str(shift_count)
    df[new_col] = shift
    return df

def preprocess_data():
    print('Loading dataset')
    # remove nrows count when saving data
    train_dataset = pd.read_csv('data/train.csv', nrows=10000)
    holidays_dataset = pd.read_csv('data/holidays_events.csv')
    item_dataset = pd.read_csv('data/items.csv')
    store_dataset = pd.read_csv('data/stores.csv')
    oil_dataset = pd.read_csv('data/oil.csv')

    min_date = min(train_dataset['date'])
    max_date = max(train_dataset['date'])
    # Create new DataFrame to take into account missing data
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    date_range[:10]

    train_df = pd.DataFrame(date_range, columns=['date'])
    train_df['unit_sales'] = 0

    train_dataset['date'] = pd.to_datetime(train_dataset['date'])
    train_df = train_df.merge(train_dataset, how='left', on='date').drop(['unit_sales_x'], axis=1).fillna(0)
    train_df.rename(columns={'unit_sales_y': 'unit_sales'}, inplace=True)

    for day in range(1,8):
        train_df = add_time_diff(train_df, 'unit_sales', day)

    # Get date features
    train_df['year'] = train_df['date'].dt.year
    train_df['day_of_year'] = train_df['date'].dt.dayofyear
    train_df['day_of_week'] = train_df['date'].dt.weekday
    train_df['week_of_year'] = train_df['date'].dt.week
    train_df['day_of_month'] = train_df['date'].dt.day
    train_df['quarter'] = train_df['date'].dt.quarter

    # Get holiday features
    holidays_dataset['date'] = pd.to_datetime(holidays_dataset['date'])
    holidays_dataset = pd.get_dummies(holidays_dataset, columns=['locale','locale_name', 'type'], prefix=['locale','locale_name', 'holiday_type'])
    holidays_dataset = holidays_dataset.drop(columns=['description','transferred'])
    train_df = train_df.merge(holidays_dataset, how='left', on='date')
    train_df.rename(columns={'type': 'holiday_type'}, inplace=True)

    # Get item features
    item_dataset = pd.get_dummies(item_dataset, columns=['family'], prefix=['family'])
    item_dataset = item_dataset.drop(columns=['class'])
    train_df = train_df.merge(item_dataset, how='left', on='item_nbr')

    # Get store features
    store_dataset['city_state'] = store_dataset[['city', 'state']].apply(lambda x: '_'.join(x), axis=1)
    store_dataset = pd.get_dummies(store_dataset, columns=['city_state', 'type'], prefix=['city_state', 'store_type'])
    store_dataset = store_dataset.drop(columns=['city', 'state'])
    train_df = train_df.merge(store_dataset, how='left', on='store_nbr')
    train_df.rename(columns={'type': 'location_type'}, inplace=True)

    # Get oil features
    oil_dataset['date'] = pd.to_datetime(oil_dataset['date'])
    oil_dataset = oil_dataset.sort_values(by=['date'], ascending=[True])
    oil_dataset.set_index('date', inplace=True)
    oil_dataset = oil_dataset.resample('D').ffill().reset_index()
    train_df = train_df.merge(oil_dataset, how='left', on='date')

    # Encoding for tree-based methods
    # DONE

    print('@create_dataset - done with preprocess')
    return train_df

dataset = preprocess_data()
print(dataset.head())
print(dataset.columns)
import numpy as np
import pandas as pd

np.random.seed(42)
n = 22000

car_names = [
    'Chevrolet Trax', 'GMC Terrain', 'Jeep Wrangler', 'Jeep Renegade',
    'Ford F-150', 'Toyota Camry', 'Honda Civic', 'Nissan Altima',
    'Hyundai Elantra', 'Kia Sportage', 'Ford Escape', 'Chevrolet Equinox',
    'Toyota RAV4', 'Honda CR-V', 'Subaru Outback', 'Mazda CX-5',
    'Dodge Charger', 'Ford Mustang', 'Chevrolet Malibu', 'Toyota Tacoma',
    'Ram 1500', 'Jeep Grand Cherokee', 'Ford Explorer', 'Toyota Highlander',
    'BMW 3 Series', 'Mercedes C-Class', 'Audi A4', 'Lexus RX',
    'Volkswagen Jetta', 'Nissan Rogue'
]

years = np.random.choice(range(2010, 2023), n, p=[
    0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10,
    0.11, 0.12, 0.13, 0.12, 0.07
])

miles = np.clip(
    (2023 - years) * np.random.normal(12000, 3000, n) + np.random.normal(0, 5000, n),
    500, 220000
).astype(int)

base_price = 8000
year_coef = 1200
mile_coef = 0.08
noise = np.random.normal(0, 2500, n)

prices = np.clip(
    base_price + year_coef * (years - 2010) - mile_coef * miles + noise,
    5000, 75000
).astype(int)

prices = (np.round(prices / 10) * 10).astype(int)
names = np.random.choice(car_names, n)

df = pd.DataFrame({'Name': names, 'Year': years, 'Miles': miles, 'Price': prices})
df.to_csv('data/raw/carvana.csv', index=False)
print(f'Dataset saved: {df.shape}')
print(df.head())
print(df.describe())

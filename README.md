# Gdg_studysession_ML_project1
1. Project Documentation
 Project Overview
This project builds a machine learning system that estimates ride prices based on trip details and contextual factors such as distance, traffic, demand, and weather conditions. The goal is to demonstrate the complete machine learning workflow from dataset cleaning to feature engineering and evaluation.
 Problem Statement
Ride prices vary depending on multiple factors. Instead of relying on fixed pricing rules, we use machine learning to learn pricing patterns from data.
 What the Model Should Learn:
 * How distance and duration affect total price.
 * How contextual factors (traffic, demand, weather) influence cost.
 * Patterns that distinguish high-cost rides (Price > 200) from low-cost rides.
   Features Used
 * distance_km: Longer trips usually cost more.
 * duration_min: Time increases cost, especially in traffic.
 * time_of_day: Rush hours may influence pricing.
 * traffic_level: Standardized levels (Low, Medium, High).
 * weather: Standardized conditions (Clear, Rainy, Stormy).
 * demand_level: Higher demand leads to surge pricing.
 * vehicle_type: Economy, Standard, or Premium.
 * high_cost: Our target classification (1 if Price > 200, else 0).
2. Data Loading & Cleaning
import pandas as pd
import numpy as np

# Step 1: Load the dataset
# Ensure 'rides.csv' is uploaded to your environment
df = pd.read_csv("rides.csv")

# Step 2: Data Cleaning & Standardizing
# Fix inconsistent casing in categorical columns
df['weather'] = df['weather'].str.strip().str.capitalize()
df['traffic_level'] = df['traffic_level'].str.strip().str.capitalize()

# Handle missing values
# Fill missing duration with the average duration
df['duration_min'] = df['duration_min'].fillna(df['duration_min'].mean())

# Fill missing traffic_level with the most common value (mode)
df['traffic_level'] = df['traffic_level'].fillna(df['traffic_level'].mode()[0])

# Step 3: Feature Engineering
# Create a binary target 'high_cost' (1 if price > 200, else 0)
df['high_cost'] = (df['ride_price'] > 200).astype(int)

# Save the cleaned version for model training
df.to_csv('rides_processed.csv', index=False)

print("Data cleaning and feature engineering complete.")

3. Exploratory Data Analysis (EDA)
# View the first few rows
print("--- Dataset Head ---")
display(df.head())

# Check data types and missing values after cleaning
print("\n--- Data Info ---")
df.info()

# Summary statistics for numerical columns
print("\n--- Statistical Summary ---")
display(df.describe())

# Check the distribution of High Cost vs Low Cost rides
print("\n--- Target Class Distribution ---")
print(df['high_cost'].value_counts(normalize=True))

Key Dataset Statistics (170 Rides)
Based on the processing of your data, here is the summary of the cleaned dataset:
 * Total Rides: 170
 * Average Distance: 12.83 km
 * Average Price: 271.42
 * High-Cost Ratio: Approximately 58\% of the rides are classified as high_cost (Price > 200).
 * Data Integrity: All 170 rows now have complete information .

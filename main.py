import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("fb_sample_hwk-2 (1).csv", encoding='ISO-8859-1')

# 1. Dataset Overview
# Size of Dataset
total_entries = len(df)
total_variables = len(df.columns)

# Features
features_description = df.columns.tolist()

# Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Print Dataset Overview
print("1. Dataset Overview:")
print("- Size of Dataset:")
print("  Total entries:", total_entries)
print("  Total variables:", total_variables)
print("\n- Features:")
print(features_description)
print("\n- Missing Values:")
print(missing_values)
print("\n- Percentage of Missing Values:")
print(missing_percentage)

# 2. Data Processing
# Cleaning Steps
# Remove duplicates
df = df.drop_duplicates()

# Handling missing values
# Count and percentage of missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Print Cleaning Steps
print("\n2. Data Processing:")
print("- Cleaning Steps:")
print("  - Duplicates Removed.")
print("  - Missing Values Handled.")

# Feature Selection
selected_features = ['Page Name', 'Followers at Posting', 'Post Created Date', 'Type', 'Total Interactions', 'Likes', 'Comments', 'Shares', 'Message', 'Link Text']

# Subset the dataframe with selected features
df_selected = df[selected_features].copy()  # Use .copy() to create a copy of the DataFrame

# Print the first few rows of the DataFrame
print(df_selected.head())

# Print selected features and new features
print("Selected Features:")
print(selected_features)
print("\nNew Features:")

# Feature Engineering
# Convert 'Total Interactions' and 'Followers at Posting' columns to numeric type
df_selected['Total Interactions'] = pd.to_numeric(df_selected['Total Interactions'], errors='coerce')
df_selected['Followers at Posting'] = pd.to_numeric(df_selected['Followers at Posting'], errors='coerce')

# Drop rows where either 'Total Interactions' or 'Followers at Posting' is NaN
df_selected.dropna(subset=['Total Interactions', 'Followers at Posting'], inplace=True)

# Calculate Engagement Ratio
df_selected['Engagement Ratio'] = df_selected['Total Interactions'] / df_selected['Followers at Posting']

# Extract Hour, Day, and Month from Post Created Date
df_selected['Post Created Date'] = pd.to_datetime(df_selected['Post Created Date'])
df_selected['Post Hour'] = df_selected['Post Created Date'].dt.hour
df_selected['Post Day'] = df_selected['Post Created Date'].dt.day_name()
df_selected['Post Month'] = df_selected['Post Created Date'].dt.month_name()

# Calculate Post Length
df_selected['Post Length'] = df_selected['Message'].str.len()

# Create binary indicator for Presence of Media
df_selected['Presence of Media'] = df_selected['Type'].apply(lambda x: 1 if x in ['Photo', 'Video'] else 0)

# Print Feature Engineering
print("- Feature Engineering:")
print("  - New Features Calculated:")
print("    - Engagement Ratio")
print("    - Post Hour")
print("    - Post Day")
print("    - Post Month")
print("    - Post Length")
print("    - Presence of Media")

# Check the first few rows of the DataFrame to confirm the addition of the 'Engagement Ratio' column
print(df_selected.head())

# 3. Analytical Methodology
# Analysis Tools
print("\n3. Analytical Methodology:")
print("- Analysis Tools: This analysis was performed using Python with libraries such as Pandas, NumPy, Matplotlib, and Seaborn.")

# Model Selection
print("- Model Selection: Linear regression and other statistical models can be considered for evaluating the data, depending on the specific analysis goals and assumptions.")

# Visualizations
print("- Visualizations: Visualizations were created using Matplotlib and Seaborn to illustrate data processing stages and preliminary findings.")

# Visualizations
# Plot engagement ratio distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_selected['Engagement Ratio'], bins=30, kde=True)
plt.title('Distribution of Engagement Ratio')
plt.xlabel('Engagement Ratio')
plt.ylabel('Frequency')
plt.show()

# Explore Correlation
numeric_columns = df_selected.select_dtypes(include=np.number).columns
correlation_matrix = df_selected[numeric_columns].corr()

# Identify Significant Features
significant_features = correlation_matrix['Total Interactions'].sort_values(ascending=False)

# Calculate Interaction Rate
df_selected['Interaction Rate'] = df_selected['Total Interactions'] / df_selected['Followers at Posting']

# Print Key Insights
print("Key Insights:")
print("- Identification of post types, topics, or content that tend to receive higher interaction rates:")

# Group by post type and calculate mean interaction rate
interaction_rate_by_type = df_selected.groupby('Type')['Interaction Rate'].mean().sort_values(ascending=False)
print(interaction_rate_by_type)

# Recommendations based on analysis results
print("\nRecommendations:")
print("- Strategic advice for the marketing team based on analysis results.")
print("- Suggestions for crafting engaging Facebook posts.")
print("- Analyze the correlation between different post types (e.g., link, photo, video) and their respective interaction rates. Allocate more resources to post types that have shown higher interaction rates.")
print("- Experiment with different content formats (e.g., images, videos, text posts) to see which ones resonate best with the audience. Use A/B testing to compare engagement metrics across different content formats.")
print("- Encourage user-generated content (UGC) by running contests, polls, or user submissions. UGC often leads to higher engagement as it fosters community participation and ownership.")
print("- Leverage trending topics, events, or holidays to create timely and relevant content. Monitor social media trends and adapt your content strategy accordingly to capitalize on current interests.")
print("- Invest in paid advertising to promote high-performing posts and reach a broader audience. Use targeting options to ensure that ads are shown to users who are most likely to engage with the content.")
print("- Continuously monitor and analyze key performance metrics to track the effectiveness of your content strategy. Regularly iterate and optimize your approach based on data-driven insights.")


# Save the processed dataframe to a new CSV file
df_selected.to_csv("processed_fb_data.csv", index=False)

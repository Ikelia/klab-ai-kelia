# Create a text summary for your report
report_text = f"""
# Weekend Assignment 2 Report

## Dataset
- Name: Tips Dataset
- Source: Seaborn (https://github.com/mwaskom/seaborn-data)
- License: BSD 3-Clause
- Shape: {df.shape}

## Question Explored
What factors influence tipping behavior at restaurants?
- Does day of week affect tip amounts?
- Does smoking status correlate with tipping?
- What is the relationship between total bill and tip?

## Key Findings
1. Weekend dinners (Saturday/Sunday) generate higher tips compared to weekday lunches
2. There's a strong positive correlation between total bill and tip amount
3. Smokers and non-smokers show similar tipping patterns
4. Average tip percentage is approximately {df['tip_percentage'].mean():.2f}%

## Limitations
1. Dataset is relatively small (244 observations)
2. Data collected from a single restaurant location
3. No information about server quality or customer satisfaction
4. Tips may be influenced by factors not captured in the data

## Charts
- Chart 1: Box plot of tips by day of week
- Chart 2: Scatter plot of total bill vs tip by smoking status
"""

# Save report
with open('../reports/weekend-a2-report.md', 'w') as f:
    f.write(report_text)

print(" Report saved to: reports/weekend-a2-report.md")

# Create reflection file
reflection_text = """
# Assignment 2 Reflection

## Which transform took the longest to get right, and why?

The NumPy vectorized computation took the longest to get right. I initially tried to use a loop to calculate the z-scores, but then I had to rethink it to use broadcasting and vectorized operations. The challenge was understanding how to apply operations to the entire array at once without explicit loops. Once I understood the concept of broadcasting, it became much easier.

## What would you do differently if you had another dataset to analyze this weekend?

If I had another dataset, I would:
1. Spend more time exploring the data visually before cleaning
2. Create more feature engineering columns based on domain knowledge
3. Try more advanced pivot tables to uncover hidden patterns
4. Document my cleaning decisions more thoroughly from the start
5. Use more automated EDA tools like pandas-profiling to speed up initial exploration
"""

# Save reflection
with open('../reports/weekend-a2-reflection.md', 'w') as f:
    f.write(reflection_text)

print(" Reflection saved to: reports/weekend-a2-reflection.md")
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Read detections from JSON file
with open('detections.json', 'r') as json_file:
    detections = json.load(json_file)

# Extract confidence levels
confidence_levels = [detection['confidence'] for detection in detections]

# Extract names and their frequencies
name_frequencies = {}
for detection in detections:
    name = detection['name']
    name_frequencies[name] = name_frequencies.get(name, 0) + 1

# Plot histogram
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.hist(confidence_levels, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Confidence Level (%)')
plt.ylabel('Frequency')
plt.title('Confidence Level Distribution')
plt.grid(True)

# Plot bar chart of name frequencies
plt.subplot(2, 3, 2)
names = list(name_frequencies.keys())
frequencies = list(name_frequencies.values())
plt.bar(names, frequencies, color='lightgreen')
plt.xlabel('Name')
plt.ylabel('Frequency')
plt.title('Frequency of Each Name')
plt.grid(True)

# Plot pie chart of name frequencies
plt.subplot(2, 3, 3)
plt.pie(frequencies, labels=names, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Distribution of Names')
plt.axis('equal')

# Plot box plot of confidence levels
plt.subplot(2, 3, 4)
plt.boxplot(confidence_levels)
plt.xlabel('')
plt.ylabel('Confidence Level (%)')
plt.title('Box Plot of Confidence Levels')

# Plot violin plot of confidence levels
plt.subplot(2, 3, 5)
plt.violinplot(confidence_levels, showmeans=True)
plt.xlabel('')
plt.ylabel('Confidence Level (%)')
plt.title('Violin Plot of Confidence Levels')

# Plot frequency polygon of confidence levels
plt.subplot(2, 3, 6)
plt.hist(confidence_levels, bins=20, color='skyblue', edgecolor='black', density=True, alpha=0.5)
kde = gaussian_kde(confidence_levels)
x_vals = sorted(confidence_levels)
plt.plot(x_vals, kde(x_vals), color='red')
plt.xlabel('Confidence Level (%)')
plt.ylabel('Density')
plt.title('Frequency Polygon of Confidence Levels')
plt.grid(True)

plt.tight_layout()
plt.show()

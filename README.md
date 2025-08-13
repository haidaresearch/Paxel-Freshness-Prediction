# Paxel-Freshness-Prediction
Freshness prediction in cold chain logistic using decision tree in Python with Colab.

## About Paxel
• Paxel is known as a pioneer of same-day delivery services in Indonesia, initially focusing on delivering home-cooked meals and MSME products. However, they are now expanding their portfolio with temperature-controlled delivery services for products such as frozen food, fresh food, and pharmaceuticals.  
• Paxel does not operate across the entire cold chain (from upstream to downstream), but is stronger in the last-mile segment. They provide solutions based on insulated boxes and refrigerated fleets.  
• One of Paxel's strengths is targeting the frozen food SME segment and direct-to-consumer (DTC) brands, which previously found it difficult to access the cold chain due to high costs or lack of flexibility.

## About Freshness Prediction
• Ensuring product quality all the way to the consumer, especially for temperature- and time-sensitive products such as frozen foods, fresh ingredients, or pharmaceuticals.  
• Shifting the paradigm from merely monitoring temperature to anticipating quality degradation (predictive).  
• Opening up monetization opportunities from data insights, not just physical logistics. For example, offering API integration for major partners such as e-grocery or meal kit services.

## Decision Tree
Like human experts who learn from experience, machines can also be trained to extract knowledge from data using algorithms such as decision trees. The accuracy of a decision tree depends on three factors:
1. the amount of training data. more data improves learning.
2. the number of variables available for decision-making. more options can lead to better outcomes.
3. the tree’s efficiency. fewer, more effective questions lead to quicker decisions.

## Data Source
In this project, the data source used is dummy data on the delivery of products via Paxel, such as fresh produce, frozen food, meat/seafood, and pharmaceuticals.
### Training Data: 80%
<pre>package_id	origin_city	destination_city	distance_km	transport_mode	avg_temp_celsius	humidity_percent	delivery_time_hours	package_type	packaging_quality	freshness_status
PAX100000	Medan	Bandung	38.9	Motorbike	10.9	63.1	49	Meat/Seafood	Good	Spoiled
PAX100001	Makassar	Bandung	190.4	Refrigerated Truck	14.5	56.3	29.1	Meat/Seafood	Good	Less Fresh
PAX100002	Bandung	Medan	987.3	Van	12.9	80.4	10.4	Frozen Food	Good	Less Fresh
PAX100003	Makassar	Makassar	1205	Van	19	39.6	12.3	Frozen Food	Excellent	Less Fresh
PAX100004	Makassar	Surabaya	1504.1	Motorbike	7.1	31.5	68.4	Pharmaceuticals	Poor	Spoiled</pre>
### Testing Data: 20%
<pre>package_id	origin_city	destination_city	distance_km	transport_mode	avg_temp_celsius	humidity_percent	delivery_time_hours	package_type	packaging_quality
PAX100800	Medan	Bandung	1080.3	Van	2.4	37.3	1.6	Fresh Produce	Good
PAX100801	Surabaya	Makassar	725.3	Van	11.9	34.7	13	Pharmaceuticals	Good
PAX100802	Makassar	Bandung	1844.6	Refrigerated Truck	24.8	61.6	42.2	Pharmaceuticals	Good
PAX100803	Medan	Surabaya	1139	Refrigerated Truck	19.3	54.2	51.7	Fresh Produce	Average
PAX100804	Bandung	Jakarta	2473.3	Refrigerated Truck	5.3	65.3	66.1	Pharmaceuticals	Good</pre>

## Python Libraries
Decision Tree using Pandas, Sklearn, PIL, Matplotlib, and Seaborn.
<pre>import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder</pre>
<pre>from sklearn.tree import export_graphviz
import graphviz
from PIL import Image
import os</pre>
<pre>import matplotlib.pyplot as plt
import seaborn as sns</pre>

## Results
<pre>distance_km	transport_mode	avg_temp_celsius	humidity_percent	delivery_time_hours	package_type	packaging_quality	freshness_status
1080.3	Van	2.4	37.3	1.6	Fresh Produce	Good	Fresh
725.3	Van	11.9	34.7	13	Pharmaceuticals	Good	Less Fresh
1844.6	Refrigerated Truck	24.8	61.6	42.2	Pharmaceuticals	Good	Spoiled
1139	Refrigerated Truck	19.3	54.2	51.7	Fresh Produce	Average	Spoiled
2473.3	Refrigerated Truck	5.3	65.3	66.1	Pharmaceuticals	Good	Spoiled</pre>
freshness_status in the previously empty testing data is now filled in.

## Feature Importance
| | Feature | Importance |
| --- | --- | --- |
| 1 | delivery_time_hours | 0.520698 |
| 2 | avg_temp_celsius | 0.237335 |
| 3 | distance_km | 0.179214 |
| 4 | packaging_quality | 0.033394 |
| 5 | package_type | 0.010974 |
| 6 | humidity_percent | 0.010659 |
| 7 | transport_mode | 0.007725 |

• delivery_time_hours has a much higher score (around 0.52) than other features. This shows that delivery time is the main predictor of freshness status, far surpassing temperature, distance, or other factors. This underscores the urgency of focusing on efficiency and speed in the delivery process, especially for goods that are susceptible to damage.  
• the high importance score for avg_temp_celsius (around 0.24) reinforces the importance of temperature monitoring during delivery. If a company does not yet have a robust temperature monitoring system, this data provides strong justification for investing in temperature sensors or other environmental monitoring solutions.  
• distance_km also has a significant importance score (around 0.18). Although not as high as delivery time, this indicates that distance traveled also contributes to the risk of freshness decline. This may be related to longer travel times or potential variations in environmental conditions during long-distance travel.

## Notes
Feature importance analysis is a key highlight from a business perspective, as it directly highlights areas that most need attention or intervention.

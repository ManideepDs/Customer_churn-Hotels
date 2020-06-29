# Hotel Bookings Customer Churn

Resorts/Hotels cannot expect all customers who book a room to visit. Few of them either cancel the booking or don't show up.
Such a scenario not only blocks the revenue to hotels but hinders them from offering a booking to a customer who can visit. 
If we study the patterns of customer behavior in the past-bookings and predict the number of customers who would cancel the
booking on a particular day, then we can oversell the rooms and make up for cancellations/No-show.

The dataset is extracted from an article written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019. 
The data set has 119,390 observations of two types of hotels – Resort hotels (H1) and City Hotel (H2). Each observation has 31 attributes. 
All data pertaining to customer or hotel identification are removed from this data set due to privacy concerns. 
Building a machine learning model using R on 119,390 observations would create huge computational overhead especially while tuning the models. 
To avoid this, we considered only City Hotels(H2) data set that has 79,326 observations for analysis. All data pertaining to Resort hotels are excluded from the analysis.

[Dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)

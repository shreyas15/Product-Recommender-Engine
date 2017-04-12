# User-browsing-model-based-Recommender-Engine

### Team: 
Shreyas S. Bhat (ssubra15@uncc.edu)

Lakshmi Udupa   (ludupa@uncc.edu)

## Proposed Project
In this project we use a large dataset of over 3.5GiB of user browsing data to model a Recommender Engine using Apache Spark.

## Data Source
We use the data available from Yahoo! Research. The data set being [Yahoo Data Targeting User Modeling, Version 1.0.](https://webscope.sandbox.yahoo.com/catalog.php?datatype=a&did=78)  

This data set contains a small sample of user profiles and their interests generated from several months of user activities at Yahoo webpages. Each user is represented as one feature vector and its associated labels, where all user identifiers were removed. Feature vectors are derived from user activities during a training period of 90 days, and labels from a test period of 2 weeks that immediately followed the training period. Each dimension of the feature vector quantifies a user activity with a certain interest category from an internal Yahoo taxonomy (e.g., "Sports/Baseball", "Travel/Europe"), calculated from user interactions with pages, ads, and search results, all of which are internally classified into these interest categories. The labels are derived in a similar way, based on user interactions with classified pages, ads, and search results during the test period. It is important to note that there exists a hierarchical structure among the labels, which is also provided in the data set.


## Business Use Case/ Advantage


## Project Objectives


## Technology / Tools Proposed
1) Amazon AWS - S3, EMR instances. 
2) Apache Spark
3) Apache Zepplin

## Installation


 

## License

MIT License
Copyright (c) 2017 Shreyas S. Bhat, Lakshmi Udupa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

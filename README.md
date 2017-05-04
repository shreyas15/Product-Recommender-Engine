# Product Recommender Engine

## Teams:
Lakshmi Udupa (800956319)

Shreyas Subramanya Bhat (800958406)

## Input files:
Cosine-similarity method:

    ratings2.csv
Alternating least squares method:

    ratings_als.csv
    metadata.csv
    Single_user_rating.txt

## Steps to execute:
### Cosine-similarity method: To execute recommenderEngine.py

1. Place the input file in the hdfs

        sudo hdfs dfs -put <file_name> /user/root/

2. To run the recommenderEngine.py

        spark-submit --driver-memory 2g recommenderEngine.py ratings2.csv 9994 0439893577 > test.out

3. To run the cos_similarity.py

        spark-submit --driver-memory 2g cos_similarity.py ratings.csv meta_data.csv 5594 > output.out 


### Alternating least square method: To execute recommenderEngine_ALS.py

1. Place the input file in the hdfs

        sudo hdfs dfs -put <file_name> /user/root/

2. To run the recommenderEngine.py

        spark-submit --driver-memory 2g recommenderEngine_ALS.py new_data.csv meta_data.csv personalRatings.txt > output.out

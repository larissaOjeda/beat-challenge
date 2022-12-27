# Fares Estimations 

Fares estimations is a Python program for calculating fares based on valid segments. 
It takes a csv file as database and it sets four columns: id_ride, latitude, longitude, timestamp. 
Two consecutive tuples are used to calculate a segment's speed, if the speed is > 100km/h then that segment is invalid. 
This program removes those invalid tuples and calulates the fare based on the speed, distance and time of the day. 
There is a Test Class provided. 

### Data

The input data to test this program is the file: 'paths.csv'. 
I have also created some other csv files one that is empty ('empty.csv').
This because the main function receives a path as a parameter,
otherwise just with an empty dataframe should've been enough. 
The file 'results.csv' corresponds to a csv file with the results of the execution of my code. 
The file 'random_empty.csv' is a copy of 'paths.csv' but it has some random empty cells (for the test). 


### Dev
I use python as the main tool for development. 
Basically I used Jupyter Notebook, and runned it there. 
In order to run the test class, I had to type at the end of it:
```python
if __name__ == "__main__": 
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
## Usage
The main function is fare_estimation(filepath) and it receives the path of the csv file to read as a parameter. 
An example to run this project is: 

```python
estimations = fare_estimation('paths.csv')

```

An expected output is a DataFrame and a csv file with 2 columns: id_ride, estimated_fare

id_ride 	fare_estimate
1 	11.34
2 	13.10
3 	32.31
4 	2.65
5 	22.78
6 	9.41
7 	29.85
8 	9.21
9 	6.35



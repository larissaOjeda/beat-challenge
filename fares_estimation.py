#!/usr/bin/env python
# coding: utf-8

# In[113]:


from datetime import datetime, time
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal 
import unittest

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """ Calculates the haversine distance given two coordintes. It is used by calculate_speed()
    This function was taken from:
        https://stackoverflow.com/questions/43450530/repeated-calculation-between-consecutive-rows-of-pandas-dataframe

    Returns: Haversine distance of given points 
    Parameters
    ----------
    lat1, lat2, lon1, lon2: float 
    to_radians: Boolean 
    earth_radius: float 
    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def check_typos(df):
    """ Check that the dataframe passed has four columns and that these columns are the indicated type 
    Returns: Boolean 
    
    Parameters
    ----------
    df : Pandas DataFrame
    """
    types = list(df.dtypes.to_numpy())
    resp = False
    if len(types) == 4:
        values = ['int64', 'float64','float64', 'int64']
        resp = types == values
    return resp

def calculate_speed(group):
    """ Calculates distance(m), time(s), speed(km/h), should_remove per passed group
    If the calculated speed is > 100 km/h, then should_remove will be True, else False.
    It serves as an auxiliary function to filter_data()
    Returns: DataFrame with the aggregated columns
        dist(m): float 
        time(s): float 
        speed(km/h): float
        should_remove: boolean

    Parameters
    ----------
    group : Pandas DataFrame
    """
    result = pd.DataFrame()
    if group.shape[0] > 1:
        group.timestamp = pd.to_datetime(group['timestamp'], unit='s')
        group['dist(m)'] = haversine(group['lat'].shift(), group['lon'].shift(), group['lat'], group['lon'], to_radians = True) * 1000
        group['time(s)'] = (group['timestamp'] - group['timestamp'].shift()).astype('timedelta64[s]').astype('Int64')
        group['speed(km/h)'] = (group['dist(m)'] * 3600) / (group['time(s)']*1000)
        group['should_remove'] = (group['id_ride'] == group['id_ride'].shift()) & (group['speed(km/h)'] > 100)
        result = group.copy(deep=True)
    return result

def filter_data(group):
    """ Removes all the rows where 'should_remove' column equals 'True'. 
    Removes 'should_remove' column as well, and it uses calculate_speed()
    Returns the passed DataFrame without the rows which should not be contemplated (> 100km/h)

    Parameters
    ----------
    group : Pandas DataFrame
    """
    speed_df = calculate_speed(group)
    if 'should_remove' in speed_df and speed_df.should_remove.any():
        clean_df = group.drop(group[group['should_remove'] == True].index)
        return filter_data(clean_df)
    return speed_df.drop(columns = ['should_remove'], errors = 'ignore')

def assign_fare_amount(group):
    """ Function that assgings the correspondant fare amount to each segment. 
    This function is used by fare_estimation()
    Returns: DataFrame with the aggregated column 'fare_amount'

    Parameters
    ----------
    group : Pandas DataFrame
    """
    limit_5 = time(hour = 5, minute = 0, second = 0)
    limit_12 = time(hour = 0, minute = 0, second = 0)
    speed_df = filter_data(group)
    time_of_day = speed_df.timestamp.dt.time
    moving_cond = speed_df['speed(km/h)'] > 10
    early_time_cond = (limit_12 < time_of_day) & (time_of_day <= limit_5)
    late_time_cond = get_ipython().getoutput('(limit_12 < time_of_day) | !(time_of_day <= limit_5)')
    conditions_fare_amt = [ 
        (moving_cond) & (early_time_cond), 
        (moving_cond) & (late_time_cond), 
        (speed_df['speed(km/h)'] <= 10) ]
    values = [1.3, 0.74, 11.9]
    speed_df['fare_amount'] = np.select(conditions_fare_amt, values)
    return speed_df

def estimate_fare_row(row):
    """ Calculates the fare based on the correspondant fare amount. 
    If 'fare_amount' == 11.9 then it will multiply by the time, 
        otherwise it will multiply the fare amount by the distance.
    This function is used by fare_estimation()

    Parameters
    ----------
    row : Pandas DataFrame row
    """
    if (row.fare_amount == 11.9):
        return row.fare_amount * (row['time(s)']/3600)
    return row.fare_amount * (row['dist(m)']/1000)

def fare_estimation(filepath):
    """ Groups by the id_ride and performs the following fuctions to each group: 
        assign_fare_amount(), calculate_estimated_fare() and calculates the fare.
        It checks the passed file: if it's empty and if the data types of columns are the correct.
        When it finishes with the logic, then sums the fare_estimation for each group. 
        
    Returns: empty DataFrame if the passed file is empty otherwise 
             returns two columned DataFrame ('id_ride', 'fare_estimate') 

    Parameters
    ----------
    filepath : path to csv file 
    """
    col_list = ['id_ride', 'lat', 'lon', 'timestamp']
    final_table = pd.DataFrame()
    df = pd.read_csv(f'{filepath}', names = col_list) 
    if df.shape[0] > 1 and check_typos(df):
        std_flag = 1.3
        estimations_df = df.groupby('id_ride').apply(lambda x: assign_fare_amount(x))
        estimations_df['fare_estimation'] = estimations_df.apply(estimate_fare_row, axis = 1)
        estimations_df.rename(columns = {'id_ride' : 'group'}, inplace = True)
        final_table = estimations_df.groupby('id_ride').agg(
            fare_estimate = pd.NamedAgg('fare_estimation', 'sum') )
        final_table.fare_estimate = round(final_table.fare_estimate + std_flag, 2)
        final_table.to_csv('results.csv')
    else: 
        print('Your dataframe must contain at least two tuples or its empty')
    return final_table


# In[114]:


#------------------------------------------------------------------------------------------------------------
# Executing the main function fare_estimation() with the given csv file 
#------------------------------------------------------------------------------------------------------------
fare_amounts = fare_estimation('paths.csv')
fare_amounts.head(9)


# In[207]:


#------------------------------------------------------------------------------------------------------------
# Here I use database_df = expected_df because I dont have the expected answers. 
#------------------------------------------------------------------------------------------------------------
# This tests could be performed creating dataframes inside each test
# The considered files are and dataframes are:
col_list = ['id_ride', 'lat', 'lon', 'timestamp']
database_df = pd.read_csv('paths.csv', names = col_list)        # original csv file 
expected_df = pd.read_csv('paths.csv', names = col_list) 
empty_df = pd.read_csv('empty.csv', names = col_list)           # an empty csv file 

data_speed = {'id_ride': ['1', '1', '1', '1'],
              'lat': [37.953066, 37.953009, 37.953195, 37.953433000000004], 
              'lon' : [23.735606,23.735592999999998, 23.736224, 23.736926], 
              'timestamp' : [1405587697, 1405587707, 1405587717, 1405587727]
             }

data = {'id_ride': ['1', '1', '1', '1'],
        'lat': [37.953066, 37.953009, 37.953195, 37.953433000000004], 
        'lon' : [23.735606,23.735592999999998, 23.736224, 23.736926], 
        'timestamp' : [1405587697, 1405587707, 1405587717, 1405587727], 
        'dist(m)' : [0, 6.439786548789833, 59.06477219524176, 66.99857255291546], 
        'time(s)' : [0, 10, 10, 10],
        'speed(km/s)' : [0, 2.31832315756434, 21.26331799028703, 24.119486119049565]
       }
data_speed_df = pd.DataFrame(data_speed)
data_df = pd.DataFrame(data)

class TestFaresEstimation(unittest.TestCase):
        
    def test_haversine(self):
        import haversine as hv
        import random
        from tqdm import tqdm
        start, end = 20, 40    
        # generate a list of random coords 
        coords = [(random.uniform(start+10, end), random.uniform(start, end-10),
                   random.uniform(start+10, end), random.uniform(start, end-10) ) for _ in range(1000)]
        for item in tqdm(coords):
            coords_a = item[0], item[1]
            coords_b = item[2], item[3]
            result = hv.haversine(coords_a, coords_b)
            expected = haversine(item[0], item[1],item[2], item[3], to_radians = True)
            self.assertAlmostEqual(result, expected, places = 4)
        print(f'haversine() test completed')
    
    def test_check_typos(self):
        # generate a random dataframe of floats and here is also considered the number 
        # of columns not matching the intented 
        df = pd.DataFrame(np.random.rand(253, 830) * 254)
        data = pd.read_csv('paths.csv')
        self.assertEqual(check_typos(df), False)
        self.assertEqual(check_typos(empty_df), False)
        self.assertEqual(check_typos(data), True)
        print(f'check_typos() test completed')
        
    def test_calculate_speed(self):
        result = calculate_speed(data_speed_df)
        empty = calculate_speed(empty_df)
        expected = [float('NaN'), 2.31832315756434, 21.26331799028703, 24.119486119049565]
        expected_round = list(np.around(expected, 4))
        result_round = list(np.around(list(result['speed(km/h)']), 4))
        self.assertEqual(result.equals(empty), False)
        self.assertAlmostEqual(result_round[1:] == expected_round[1:], True, places = 8)
        print(f'calculate_speed() test completed')
        
    def test_filter_data(self):
        expected_cols = ['id_ride', 'lat', 'lon', 'timestamp', 'dist(m)', 'time(s)', 'speed(km/h)']
        filter_db = filter_data(database_df)
        filter_db_cols =  list(filter_db.columns)
        empty_cols = list(filter_data(empty_df).columns)
        self.assertEqual((expected_cols == filter_db_cols), True)
        self.assertEqual((expected_cols == empty_cols), False)
        self.assertEqual((filter_db['speed(km/h)'] > 100).any(), False)  #check if there is a value bigger than 100
        print(f'filter_data() test completed')
    
    def test_assign_fare_amount(self):
        fares = assign_fare_amount(data_df)
        expected = [0, 11.90, 0.74, 0.74]
        expected_2 = [0.74, 0.74, 0.74, 0.74]
        expected_3 = [11.90, 11.90, 11.90, 11.90]
        self.assertEqual((list(fares['fare_amount']) == expected), True)
        self.assertEqual((list(fares['fare_amount']) == expected_2), False)
        self.assertEqual((list(fares['fare_amount']) == expected_3), False)
        print(f'assign_faretest() completed')
    
    def test_estimate_fare_row(self):
        fares = assign_fare_amount(data_df)
        fares['fare_estimation'] = fares.apply(estimate_fare_row, axis = 1)
        values = [0, (11.90 * (10/3600)) , (0.74* ((59.07)/1000)), (0.74*(67/1000))]
        fares_round = list(np.around(values, 4))
        values_round = list(np.around(list(fares['fare_estimation']), 4))
        self.assertAlmostEqual(fares_round[1:] == values_round[1:], True,  places=7)
        print(f'estimate_fare() test completed')
        
    def test_fare_estimation(self): 
        result = fare_estimation('paths.csv')
        empty = fare_estimation('empty.csv')
        expected = fare_estimation('paths.csv')
        self.assertEqual(result.equals(empty), False)
        self.assertEqual(result.equals(expected), True)
        
    if __name__ == "__main__": 
        unittest.main(argv=['first-arg-is-ignored'], exit=False)


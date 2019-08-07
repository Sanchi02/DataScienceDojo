import pandas as pd
import numpy as np
pd.set_option('max_rows', 5)

# # Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
# fruits = {'Apples': pd.Series([30]), 'Bananas': pd.Series([21])}
# fruits = pd.DataFrame(fruits)

# fruits


# # Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
# fruit_sales = {'Apples': pd.Series([35,41], index=['2017 Sales', '2018 Sales']), 'Bananas': pd.Series([21,34], index=['2017 Sales', '2018 Sales'])}
# fruit_sales = pd.DataFrame(fruit_sales)

# fruit_sales


# ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')

# ingredients

fruits = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 
                                  'Parrot', 'Parrot',
                                  'Cat','Cat','Cat'],
                   'Max Speed': [380., 370., 24., 26.,12,14,np.NaN]})

fruits.fillna(fruits.groupby('Animal').transform('mean'), inplace=True)
# fruits2 = fruits.groupby('Animal')
# print(fruits2.mean())

print(fruits)
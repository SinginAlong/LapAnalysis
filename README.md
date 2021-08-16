# LapAnalysis
Use Harry's Lap Timer data to analyze lap performance, with special focus on the influence of speed in the corners

## Current Design
Program reads in GPS point data from lap timer.
For each lap it tries to find the corner speeds.
Corner speeds are local minimums, with some filtering to avoid false positives from noise.
After all corner points have been identified, the points are categorized into labeled corners
using k-means clustering.


Now we have a name of each corner and points associated it them.
We filter the corners by number of points to remove more noise.
Then we use the corner speeds to train a linear regression model.
The coefficients for each of the corners are reported back and used to create a plot.

## Data
Data is from GPS point data from Harry's Lap Timer.
The data was converted using Excel.
This process must be repeated for new data.
In future, I hope to write a script to automatically convert the data.

## Areas for improvement
 - write a script to automatically import harry's data from hlptrl (XML format) to the csv file
 - re-name poorly named columns
 - train the model multiple times, take coefficient averages or scale each batch to fit
 - use corner combos as additional features (speed of corner 1 + speed of corner 2, 2 + 3, or  1 x 2 and so on)
 - use max speed as additional features
 - try on different data sets, either for different drivers or different courses


## Notes
 - The data presented in the example, and used to write the code is from lap times at Laguna Seca.
 - All laps in a single csv are assumed to be the same car and driver.
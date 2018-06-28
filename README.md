# Hackathon
This was an attempt at using machine learning tools to match sensor recorded data (accelerometer and magnometer) to tracker recorded locations in order to obtain locations from the sensor readings.
Time syncing was done manually based on a specific motion used as a sync signal.
Data was then cleaned from nan values and smoothed to reduce noise.
Features were then extracted from the data to be used as input for a possible future machine learning implementation,
proposed implementations should be able to time sensitive, therefore perhaps rnns, lstms or attention based networks would be suitable

import random as rd
import csv

batch = list()

# Generate batch
for i in range(765293):

    # Broken?
    if rd.random() <= 0.26:

        batch.append('0')

    else:

        batch.append('1')

# Write to csv
with open('batch.csv', 'w') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')

    writer.writerows(batch)
import pandas as pd

# Read the original CSV
df = pd.read_csv('syn.csv')

# Save as compressed CSV (GZIP)
df.to_csv('syn.csv.gz', index=False, compression='gzip')

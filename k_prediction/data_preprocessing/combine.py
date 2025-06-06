import pandas as pd

# read
combine1 = pd.read_excel("combine1.xlsx")
combine2 = pd.read_excel("combine2.xlsx")

# Get all unique column names, keeping the order: first combine2's columns, then add the newly added columns in combine1
all_columns = list(dict.fromkeys(combine2.columns.tolist() + combine1.columns.tolist()))

# reindex
combine1_aligned = combine1.reindex(columns=all_columns)
combine2_aligned = combine2.reindex(columns=all_columns)

# combine
combined = pd.concat([combine2_aligned, combine1_aligned], ignore_index=True)

# save
combined.to_excel("combined_output.xlsx", index=False)

print("done combined_output.xlsx")

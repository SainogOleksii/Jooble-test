import pandas as pd

from data_preprocess import VacancyDataFrame, Kit2Preprocessor

# upload data
test = pd.read_csv("./data/test.tsv", sep="\t", dtype={"id_job": str}, index_col="id_job")

# define batch size
batch_size = 10

# create vacancy dataframe object
vdf = VacancyDataFrame(test, batch_size=batch_size)

# split dataframe into kits
vdf = vdf.split_into_kits()

# preprocess kit with id = 2
vdf = vdf.preprocess_kit(Kit2Preprocessor(), kit_id=2)

# return preprocess dataframe
result = vdf.get_kit(kit_id=2)

# save result to file
result.to_csv("./data/test_proc.tsv", sep="\t")

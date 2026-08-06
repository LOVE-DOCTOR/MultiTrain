# Example datasets

The CSV files in this directory keep the examples and integration tests reproducible without downloading data during a test run.

## Palmer Penguins

- File: `penguins.csv`
- Source: [seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)
- Rows: 344
- Target: `species`
- Purpose: multiclass classification with categorical columns and missing values
- SHA-256: `e07636bd8af74260099ea2f8678e2eabbf35def579940cc76f67061ee16c06c1`

## Red Wine Quality

- File: `winequality-red.csv`
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Rows: 1,599
- Target: `quality`
- Purpose: regression with differently scaled continuous features
- Delimiter: semicolon (`;`)
- SHA-256: `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e`

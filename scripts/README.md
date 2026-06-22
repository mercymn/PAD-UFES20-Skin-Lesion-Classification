# Data Preparation Pipeline

## Script

To run the script simply run the following from the base of this repository.

```bash
python scripts/dataset.py
```

This will do 2 things.

1. It will download and unzip all the data under the data directory.

2. It will create a `metadata.pickle` which you can then use to import your `DataFrame`.

## Usage In Notebooks

Before you can use your notebook please run the above script.

Then if all you need is a `DataFrame` simply copy the `metadata.pickle` into your notebook.

And add the following code to your notebook:

> [!WARNING]
> For this to work the `pandas` version of your repository and the notebook must match.

```ipynb
!pip install pandas==2.3.3
import pandas as pd
pd.read_pickle("metadata.pickle")
```

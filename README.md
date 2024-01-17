# DyART

Different versions of dynamic autoregressive tree's with the option to include exogenous variables to forecast. Includes model tuning and retraining on the basis of changepoints

## Installation

Clone this repository with an IDE such as visual studio code.

Note: the code is made using python 3.11

Set up a virtual env through the command: 
```bash
python3 -m venv .venv
```

Install all of the necessary pacakges with:
```bash
pip install -r requirements.txt
```
## Usage

The first step is to download data, using the notebook data_cleaning, or import your own data into the folder ..//Data

After that the changepoints can either be found initially or using online methodology in the main script.

To run the main script, just enter the following code:

```python
py Combined_code\main.py
```

The specific models to be run can be altered. Keep in mind that it will take well over an hour to run all the different models for the entire time series.
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

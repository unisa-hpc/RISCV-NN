import pandas as pd
import json


class TimerStatsParser:
    def __init__(self, json_file: str):
        self.json_file = json_file
        with open(self.json_file) as f:
            self.json = json.load(f)

    def get_df(self) -> pd.DataFrame:
        """
        Get the data as a pandas dataframe.
        """
        df = pd.DataFrame({
            'name': self.json['name'],
            'N': self.json['pairs']['N'],
            'unroll_factor': self.json['pairs']['unroll_factor'],
            'samples': self.json['samples'],
            'data_point': self.json['data'],  # expand data points
            'average': self.json['average'],
            'median': self.json['median'],
            'variance': self.json['variance'],
            'max': self.json['max'],
            'min': self.json['min']
        })
        return df

    def concat_with(self, other) -> pd.DataFrame:
        """
        Concatenate the dataframes of the two objects.
        """
        return pd.concat([self.get_df(), other.get_df()], ignore_index=True)

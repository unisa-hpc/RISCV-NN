import pandas as pd
import json


class TimerStatsParser:
    def __init__(self, json_file: str, parse_pairs_func=lambda pairs: {}):
        self.pairs_parser = parse_pairs_func
        self.json_file = json_file
        with open(self.json_file) as f:
            self.json = json.load(f)

    def get_df(self) -> pd.DataFrame:
        """
        Get the data as a pandas dataframe.
        """
        base_dict = {
            'name': self.json['name'],
            'samples': self.json['samples'],
            'data_point': self.json['data'],  # expand data points
            'average': self.json['average'],
            'median': self.json['median'],
            'variance': self.json['variance'],
            'max': self.json['max'],
            'min': self.json['min']
        }

        # Update the base dictionary with the key-value pairs returned by the lambda function
        base_dict.update(self.pairs_parser(self.json['pairs']))
        df = pd.DataFrame(base_dict)
        return df

    def concat_with(self, other) -> pd.DataFrame:
        """
        Concatenate the dataframes of the two objects.
        """
        return pd.concat([self.get_df(), other.get_df()], ignore_index=True)

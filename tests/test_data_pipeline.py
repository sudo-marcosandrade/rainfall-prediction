import unittest

import pandas as pd

from src.data import prepare_dataset


class DataPipelineTest(unittest.TestCase):
    def test_prepare_dataset_removes_target_leakage(self):
        raw_data = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02"],
                "Location": ["Sydney", "Sydney"],
                "RainToday": ["No", "Yes"],
                "RISK_MM": [0.0, 12.4],
                "RainTomorrow": ["No", "Yes"],
            }
        )

        x, y = prepare_dataset(raw_data)

        self.assertNotIn("RISK_MM", x.columns)
        self.assertNotIn("RainTomorrow", x.columns)
        self.assertIn("Year", x.columns)
        self.assertEqual(y.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()

import unittest
from FE_Stock.FE_Quandl.FE_Quandl import FE_Quandl
from setup import quandl_api_key_file


class TestQuandlMethod(unittest.TestCase):
    """Integration test with qunadl"""
    def setUp(self):
        self.quandl_obj = FE_Quandl(quandl_api_key_file)
        self.test_stock_sym = "MSFT"

    def test_key_exists(self):
        self.assertTrue(self.quandl_obj.get_qunadl_key() is not None)

    def test_get(self):
        data = self.quandl_obj.get(self.test_stock_sym)
        self.assertEqual('Date', data.index.name)

        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in columns:
            self.assertTrue(col in data.columns)

if __name__ == '__main__':
    unittest.main()
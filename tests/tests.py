import unittest

class TestTrades(unittest.TestCase):
    def setUp(self) -> None:
        # TODO
        return super().setUp()
    
    def test_trades_no_comments(self):
        # buy(1), buy(2), sell(1), sell(2)
        # test only one trade, test avg cost and pnl
        self.assertEqual(True, True)

    def test_cash_update(self):
        # buy(1), test cash update (buying new)
        # buy(1), test cash update (adding to trade)
        # sell(1), test cash update (selling partial)
        # sell(1), test cash update (close trade)
        self.assertNotEqual(True, False)

if __name__ == "__main__":
    unittest.main(verbosity=2)
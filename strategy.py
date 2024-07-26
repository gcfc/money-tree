# class Strategy:
#     def __init__(self) -> None:
#         pass
    
#     def initialize(self):
#         pass
    
#     def next(self):
#         pass


# class Portfolio:
#     def __init__(self) -> None:
#         pass
    
#     def initialize(self):
#         pass
    
#     def next(self):
#         pass

# class Trade:
#     def __init__(self, portfolio, num_shares, entry_price, exit_price=None, stop_loss=None) -> None:
#         self.portfolio = portfolio
#         self.num_shares = num_shares
        
    
#     def close(self):
#         """Place new `Order` to close `portion` of the trade at next market price."""
#         size = copysign(max(1, round(abs(self.num_shares))), -self.num_shares)
#         order = Order(self.portfolio, size, parent_trade=self)
#         self.portfolio.orders.insert(0, order)
    


# class PriceOrPercentage:
#     def __init__(self, amount):
#         self.amount = amount
#         self.validate()
    
#     def validate(self):
#         assert isinstance(self.amount, str), "PriceOrPercentage amount should be a string."
#         validate_string = self.amount
#         if self.amount.endswith("%"):
#             validate_string = validate_string.removesuffix("%")
#         try:
#             float(validate_string)
#         except ValueError: 
#             raise ValueError(f"PriceOrPercentage amount '{self.amount}' in wrong format.")
        
# class Decision:
#     def __init__(self) -> None: 
#         self.action = Actions.HOLD
#         self.num_shares = 0
#         self.stop_loss = None
#         self.profit_target = None
        
# class TestStrategy(StrategyBase):
#     pass

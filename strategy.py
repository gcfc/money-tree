from enum import Enum, unique

class StrategyBase:
    def __init__(self) -> None:
        self.highest_price = None
        self.lowest_price = None
    
    def process(self, data):
        pass
        
@unique
class Actions(Enum):
    HOLD = 1
    BUY = 2
    SELL = 3

class PriceOrPercentage:
    def __init__(self, amount):
        self.amount = amount
        self.validate()
    
    def validate(self):
        assert isinstance(self.amount, str), "PriceOrPercentage amount should be a string."
        validate_string = self.amount
        if self.amount.endswith("%"):
            validate_string = validate_string.removesuffix("%")
        try:
            float(validate_string)
        except ValueError: 
            raise ValueError(f"PriceOrPercentage amount '{self.amount}' in wrong format.")
        
class Decision:
    def __init__(self) -> None: 
        self.action = Actions.HOLD
        self.num_shares = 0
        self.stop_loss = None
        self.profit_target = None
        
class TestStrategy(StrategyBase):
    def 
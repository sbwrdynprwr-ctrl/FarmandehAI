from dataclasses import dataclass

@dataclass(frozen=True)
class RiskManager:
    risk_per_trade: float = 0.01

    def position_size(self, equity: float, entry: float, stop: float) -> float:
        if equity <= 0 or not 0 < self.risk_per_trade <= 1:
            raise ValueError("Invalid equity or risk_per_trade")
        distance = abs(entry - stop)
        if distance <= 0:
            raise ValueError("Entry and stop must differ")
        return equity * self.risk_per_trade / distance

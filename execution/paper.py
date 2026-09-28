class PaperExecutor:
    def __init__(self, live=False, real=False):
        if live or real:
            raise RuntimeError("Safety lock: LIVE/REAL must remain OFF")
        self.orders=[]
    def submit(self, order):
        self.orders.append(dict(order, mode="PAPER"))
        return self.orders[-1]

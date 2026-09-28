import pytest
from execution.paper import PaperExecutor

def test_paper_executor_rejects_live():
    with pytest.raises(RuntimeError): PaperExecutor(live=True)
    with pytest.raises(RuntimeError): PaperExecutor(real=True)

def test_paper_executor_only_paper():
    ex=PaperExecutor(); result=ex.submit({'symbol':'GBP/USD','side':'LONG','qty':1}); assert result['mode']=='PAPER'

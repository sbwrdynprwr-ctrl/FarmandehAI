FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PAPER=True
ENV LIVE=False
ENV REAL=False
ENV NO_LOOKAHEAD=True
ENV CLOSED_ONLY=True

CMD ["sh", "-c", "python -c \"import urllib.request,os; files=['backtest/engine.py','backtest/research.py','backtest/walk_forward.py','holdout_runner.py']; [(os.makedirs(os.path.dirname(f),exist_ok=True) if os.path.dirname(f) else None,open(f,'wb').write(urllib.request.urlopen('https://raw.githubusercontent.com/sbwrdynprwr-ctrl/FarmandehAI/main/'+f,timeout=60).read())) for f in files]; print('LATEST_GITHUB_CODE_FETCHED',flush=True)\" && python -u holdout_runner.py"]

# Railway holdout validation entrypoint
# PAPER=True LIVE=False REAL=False

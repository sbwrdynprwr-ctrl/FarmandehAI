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

CMD ["python", "-u", "main.py", "--days", "60", "--folds", "4", "--research", "--serve"]

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

CMD ["python", "-u", "holdout_runner.py"]

# holdout research runner remains the deployment entrypoint
# trigger latest-commit deployment for independent holdout validation
# railway latest-main trigger
# independent holdout validation trigger 2026-10-01
# deployment verification trigger 2026-10-01
# final holdout autodeploy trigger 2026-10-01

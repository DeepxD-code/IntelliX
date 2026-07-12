FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

COPY . .

RUN mkdir -p models logs

# Train and persist a model AT BUILD TIME, not at request-serving time.
# Without this, app.py's init_backend() finds no saved model on a fresh
# container and falls back to training one in-band on the first request --
# which means a full GridSearchCV sweep (n_samples=3000, tune=true per
# config.yaml) blocking the app from answering even /api/health.
# Verified: that training does not finish within 90 seconds. Most platforms'
# health-check timeouts are far shorter than that, so an un-baked image can
# fail to ever go live. Paying this cost once here, during the build (where
# a multi-minute step is normal and expected), means the container is ready
# to serve traffic immediately on start.
RUN python profiler_main.py

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]

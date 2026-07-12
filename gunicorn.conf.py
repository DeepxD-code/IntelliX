"""Gunicorn configuration for IntelliProfile.

WORKERS DEFAULTS TO 1, ON PURPOSE.
----------------------------------
flask-limiter's default in-memory rate-limit storage is per-process. Each
gunicorn WORKER PROCESS would keep its own independent counter, silently
multiplying the configured rate limit by the worker count. THREADS within
a single process share memory, so they don't have this problem.

Single worker + multiple threads is therefore the default that makes
server.max_requests_per_minute (in config.yaml) mean what it says, with no
extra infrastructure required. It's appropriate for moderate traffic on a
single instance.

If you need to scale beyond one process (more workers, or multiple
machine instances behind a load balancer), you MUST set
RATELIMIT_STORAGE_URI to a shared backend (e.g. redis://...) before
increasing WEB_CONCURRENCY -- otherwise the original bug comes back.
"""

import os

workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("WEB_THREADS", "4"))
worker_class = "gthread"

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
timeout = int(os.environ.get("WEB_TIMEOUT", "120"))

# Log to stdout/stderr so the JSON-formatted application logs and
# gunicorn's own access/error logs both end up wherever the deployment
# platform collects container output, rather than to a file that may not
# persist or get rotated.
accesslog = "-"
errorlog = "-"

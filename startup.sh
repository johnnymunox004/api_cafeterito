#!/bin/bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --timeout 600
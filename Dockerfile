# Fix #14: use slim variant — reduces image size from ~900MB to ~125MB
FROM python:3.10-slim

WORKDIR /app

COPY . .

# Fix #13: only install production deps, not dev/test deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

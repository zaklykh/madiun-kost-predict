FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput
RUN python manage.py migrate

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "streamlit.wsgi:application"]
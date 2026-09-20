FROM python:3.13.7

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY results/ results/
COPY app.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

#Docker internal app workyng check
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

#App start command
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
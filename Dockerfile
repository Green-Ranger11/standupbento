FROM nikolaik/python-nodejs:latest
EXPOSE 8501
WORKDIR /app/standupman
RUN npm install && npm run
WORKDIR /app/client
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run app.py
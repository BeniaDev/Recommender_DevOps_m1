FROM python:3.9
COPY . ./app
WORKDIR /app/
COPY ./requirements.txt ./requirements.txt
EXPOSE 5000
RUN pip3 install --no-cache-dir -r requirements.txt
CMD [ "python3", "api.py" ]
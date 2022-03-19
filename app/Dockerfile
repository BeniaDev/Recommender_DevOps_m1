FROM python:3.9
COPY ./service ./app/service/
WORKDIR ./app/service/
EXPOSE 5000
RUN pip3 install --no-cache-dir -r requirements.txt
CMD [ "python3", "api.py" ]
#docker build --no-cache -t deploy:cgan .
#docker run --rm -p 80:80 deploy:cgan
FROM ubuntu:20.04
COPY . .

RUN apt-get update && apt-get install -y git && \
	apt update && apt -y install python3-pip && \
	pip install -r requirements.txt && \
	rm requirements.txt
EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
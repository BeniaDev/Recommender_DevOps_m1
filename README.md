# Recommender_DevOps_m1

## Creds:
* Eugene Borisov, 971901 HITS, email: evgenyboriso@gmail.com

## "How to"

### Train Model:
* cli enabled :) =>
```
foo@bar:~$ python3 model.py ----dataset=/path/to/train/dataset
```

### Evaluate:
```
foo@bar:~$ python model.py evaluate --dataset=/path/to/evaluation/dataset
```


###Deployment:
1. With docker in progress to add docker image to registry
2. Withour docker clone this repo and use ./app/


### Test API of Flask App from api.py
* http://127.0.0.1:5001/api/predict?user_id=100&M=30
* http://127.0.0.1:5001/api/log
* http://127.0.0.1:5001/api/info
* http://127.0.0.1:5001/api/reload


# Progress: 
- [x] Matrix baseline
- [x] CLI application
- [x] Worked Flask app with API
- [x] Dockerfile with docker-compose.yaml
- [ ] Little code refactoring
- [ ] CI/CD part
- [ ] Better Recommender Model 


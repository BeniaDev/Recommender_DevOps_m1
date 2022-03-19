import logging

import typer

from typing import Optional
from pathlib import Path
from als_recommender import ALSRecommender

logging.basicConfig(filename='../logs/app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

app = typer.Typer()
als_rec = ALSRecommender()


@app.command()
def train(dataset: Optional[Path] = Path("../data/ratings_train.dat")):
    als_rec.train(dataset)


@app.command()
def evaluate(dataset: Optional[Path] = Path("../data/ratings_test.dat")):
    als_rec.evaluate(dataset)


@app.command()
def predict(user_id: Optional[str], M: Optional[str]=10):
    movies, preds = als_rec.recommend(user_id, m=M)
    print(f"Recommended Films: {[int(x) for x in movies]} \n with ratings: {[str(x) for x in preds]}")

    return movies, preds


@app.command()
def reload():
    asl_rec = als_rec.warmup()


if __name__ == '__main__':
    app()

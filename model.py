import logging

import typer

from typing import Optional
from pathlib import Path
from src.als_recommender import ALSRecommender

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

app = typer.Typer()
asl_rec = ALSRecommender()


@app.command()
def train(dataset: Optional[Path] = Path("./data/ratings_train.dat")):
    asl_rec.train(dataset)


@app.command()
def evaluate(dataset: Optional[Path] = Path("./data/ratings_test.dat")):
    asl_rec.evaluate(dataset)


@app.command()
def predict(user_id: Optional[str]):
    movies, preds = asl_rec.recommend(user_id)
    print(f"Recommended Films: {[int(x) for x in movies]} \n with ratings: {[str(x) for x in preds]}")


if __name__ == '__main__':
    app()

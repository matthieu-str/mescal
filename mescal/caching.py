import pickle
from pathlib import Path
from .filesystem_constants import DIR_DATABASE_CACHE


def cache_database(database: list[dict], database_name: str) -> None:
    """
    Create a pickle file to store the database

    :param database: database to store
    :param database_name: name of the database
    :return: None
    """
    Path(DIR_DATABASE_CACHE).mkdir(parents=True, exist_ok=True)
    with open(DIR_DATABASE_CACHE / f"{database_name}.pickle", "wb") as output_file:
        pickle.dump(database, output_file)
        print(f"{database_name}.pickle created!")


def load_db(database_name: str, filepath: str = None) -> list[dict]:
    """
    Load a database from a pickle file

    :param database_name: name of the database
    :param filepath: path to the pickle file
    :return: database
    """
    if filepath is None:
        filepath = DIR_DATABASE_CACHE / f"{database_name}.pickle"

    with open(filepath, "rb") as input_file:
        db = pickle.load(input_file)

    for ds in db:
        if "categories" in ds:
            del ds["categories"]

    return db

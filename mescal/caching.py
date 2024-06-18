import pickle
from pathlib import Path

DIR_DATABASE_CACHE = Path.cwd() / "export" / "cache"


def cache_database(database: list[dict], database_name: str) -> None:
    """
    Create a pickle file to store the database
    :param database: (list of dict) database to store
    :param database_name: (str) name of the database
    :return: None
    """
    Path(DIR_DATABASE_CACHE).mkdir(parents=True, exist_ok=True)
    with open(DIR_DATABASE_CACHE / f"{database_name}.pickle", "wb") as output_file:
        pickle.dump(database, output_file)
        print(f"{database_name}.pickle created!")


def load_db(database_name: str, filepath: str = None) -> list[dict]:
    """
    Load a database from a pickle file
    :param database_name: (str) name of the database
    :param filepath: (str) path to the pickle file
    :return: (list of dict) database
    """
    if filepath is None:
        filepath = DIR_DATABASE_CACHE / f"{database_name}.pickle"

    with open(filepath, "rb") as input_file:
        db = pickle.load(input_file)

    for ds in db:
        if "categories" in ds:
            del ds["categories"]

    return db

from src.models import train_models
from src.data import FARM_LIST


def main():
    models = train_models(FARM_LIST)
    return models


if __name__ == '__main__':
    main()

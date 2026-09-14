from hycseq.engine import run_experiment
from hycseq.settings import parse_settings


if __name__ == "__main__":
    run_experiment(parse_settings())

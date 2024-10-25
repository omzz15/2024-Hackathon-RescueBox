import argparse

from flask_ml.flask_ml_cli import MLCli
from .server import server


def main():
    parser = argparse.ArgumentParser(description="Cluster images by their setting")
    cli = MLCli(server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()

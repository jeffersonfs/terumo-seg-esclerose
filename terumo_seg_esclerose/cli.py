"""CLI interface for terumo_seg_esclerose project.
"""
import typer

from .utils.get_original_data import get_datasets

app = typer.Typer()


@app.command()
def make_datasets():
    get_datasets()


@app.command()
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m terumo_seg_esclerose` and `$ terumo_seg_esclerose `.
    """

    print("Command principal")

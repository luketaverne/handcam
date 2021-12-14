import click

from handcam import env_test


@click.group()
def cli():
    pass


@cli.command()
def verify_env():
    env_test.test()
    click.echo("Environment looks good.")


@cli.command()
def dropdb():
    click.echo("Dropped the database")

import click


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
def register():
    pass


@cli.command()
def match():
    pass


if __name__ == "__main__":
    cli()

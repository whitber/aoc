import importlib
from pathlib import Path

import click
from aocd import submit
from jinja2 import Template
from loguru import logger


def get_script(year, day):
    script_name = f"waoc.{year}.day{day:02}"
    return importlib.import_module(script_name)


@click.command()
@click.option('-y', '--year', required=True, type=int)
@click.option('-d', '--day', required=True, type=int)
@click.option('-p', '--part', type=click.Choice(['a', 'b']))
@click.option('-t', '--test', is_flag=True)
@click.option('-r', '--run', is_flag=True)
@click.option('-n', '--new', is_flag=True)
@click.option('-s', '--send', is_flag=True)
def cli(year, day, part=None, test=False, run=False, new=False, send=False):
    if new:
        base_dir = Path(__file__).parent
        year_folder = base_dir / str(year)
        year_folder.mkdir(parents=True, exist_ok=True)

        template_path = base_dir / "template.py"
        if not template_path.exists():
            click.echo(f"Error: Template file '{template_path}' does not exist.")
            return

        filename = f"day{day:02}.py"
        file_path = year_folder / f"{filename}"

        # Load and render the template
        with template_path.open() as f:
            template = Template(f.read())
            content = template.render(year=year, day=day)

        if file_path.exists():
            click.echo(f"File {file_path} already exists!")
        else:
            file_path.write_text(content)
            click.echo(f"Created file: {file_path}")

    elif run:
        wrun(year, day, part=part, test=False)
    elif test:
        wrun(year, day, part=part, test=True)

    elif send:
        wsubmit(year, day, part)


def wrun(year, day, part=None, test=False):
    script = get_script(year, day)

    if part == 'a' or part is None:
        ans = script.solve(test=test, parta=True)
        print("Part a:", ans)

    if part == 'b' or part is None:
        ans = script.solve(test=test, parta=False)
        print("Part b:", ans)


def wsubmit(year, day, part):
    script = get_script(year, day)

    ans = None
    if part == 'a':
        ans = script.solve(parta=True)
    elif part == 'b':
        ans = script.solve(parta=False)

    if ans is None:
        logger.warning("No answer computed.")
        return

    submit(answer=ans, part=part, day=day, year=year)


if __name__ == "__main__":
    cli(year=2024, day=1)

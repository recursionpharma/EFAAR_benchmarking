from cookiecutter.main import cookiecutter
from roadie.constants import FLAKE8_STYLE, LINE_LENGTH
import sys


def main():

    flake8_configuration = ''
    for k, v in FLAKE8_STYLE.items():
        flake8_configuration += f'{k} = {str(v)}\n'

    cookiecutter_context = {
        'repo_name': "test-package",
        'description': "words go here",
        'type': 'package',
        'flake8': flake8_configuration,
        'line_length': LINE_LENGTH,
        'cli': sys.argv[1],
    }
    cookiecutter('..', no_input=True, extra_context=cookiecutter_context)


main()

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_schema_template_is_not_gitignored():
    rules = (ROOT / '.gitignore').read_text().splitlines()
    assert '!inputs/experiment_schema.csv' in rules  # otherwise CI cannot find the template
    assert (ROOT / 'inputs' / 'experiment_schema.csv').is_file()


def test_readme_images_exist():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for name in ('architecture.svg', 'ai-pipeline.svg', 'dashboard.svg'):
        assert f'docs/images/{name}' in readme
        assert (ROOT / 'docs' / 'images' / name).is_file()

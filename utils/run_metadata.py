"""Write a readable startup record for one optimization experiment."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess


PACKAGE_NAMES = (
    'finesse', 'torch', 'torch-geometric', 'scikit-opt',
    'numpy', 'scipy', 'networkx', 'h5py', 'tqdm', 'matplotlib',
)


def _git_info(repo_root):
    """Report unknown fields as None when Git or repository history is absent."""
    def query(*args):
        try:
            return subprocess.run(
                ['git', '-C', str(repo_root), *args],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = query('rev-parse', 'HEAD')
    status = query('status', '--porcelain', '--untracked-files=normal')
    return {
        'commit': commit,
        # Ignored experiment outputs do not count as code changes.
        'dirty': bool(status) if status is not None else None,
    }


def _software_versions():
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'packages': packages}


def _run_readme(metadata):
    """Render the same startup snapshot as a browsable Markdown record."""
    settings = metadata['settings']
    experiment = settings['experiment']
    surrogate = settings['surrogate_type']
    lines = [
        f"# {surrogate} run{experiment['run_number']}",
        '',
        f"Started (UTC): {metadata['started_at_utc']}",
        '',
        f"Optimize ITM/ETM mirror radii of curvature using a {surrogate} surrogate "
        f"and PSO for {experiment['planned_rounds']} planned feedback rounds. "
        'Generate initial Finesse graphs, train the base model, then repeat '
        'surrogate search, Finesse evaluation, and model fine-tuning.',
        '',
        'This is a startup snapshot, not a completion report. '
        'See [metadata.json](metadata.json) for the machine-readable record.',
        '',
        '## Run notes',
        '',
        experiment.get('notes') or 'No run notes provided.',
        '',
    ]

    def cell(value):
        if value is None:
            value = 'null'
        elif not isinstance(value, str):
            value = json.dumps(value, allow_nan=False)
        return (value.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('|', '&#124;')
                .replace('\r', '').replace('\n', '<br>'))

    def section(title, values, level=2):
        lines.extend([f"{'#' * level} {title}", ''])
        scalar_items = [(key, value) for key, value in values.items() if not isinstance(value, dict)]
        if scalar_items:
            lines.extend(['| Setting | Value |', '| --- | --- |'])
            lines.extend(f'| {cell(key)} | {cell(value)} |' for key, value in scalar_items)
            lines.append('')
        for key, value in values.items():
            if isinstance(value, dict):
                section(key.replace('_', ' ').capitalize(), value, level + 1)

    section('Settings', settings)
    lines.extend(['## Output paths', '', 'Paths are relative to the repository root.', ''])
    section('Files and filename templates', metadata['files'], level=3)
    section('Code revision', metadata['git'])
    section('Software versions', metadata['software'])
    lines.extend([
        'Unavailable Git information or package versions are recorded as `null`. '
        'The Git dirty flag records whether local changes existed; it does not preserve them. '
        'This snapshot does not enable resume or guarantee identical replay.',
        '',
    ])
    return '\n'.join(lines)


def write_run_metadata(path, *, repo_root, settings, paths):
    """Save metadata.json and a sibling README.md at startup, without overwriting.

    Paths are relative to repo_root, including any filename templates. This is
    provenance, not a resume checkpoint or a guarantee of deterministic replay.
    """
    path = Path(path)
    readme_path = path.with_name('README.md')
    for output in (path, readme_path):
        if output.exists():
            raise FileExistsError(f'Run record already exists: {output}')
    repo_root = Path(repo_root).resolve()
    metadata = {
        'schema_version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'settings': settings,
        'git': _git_info(repo_root),
        'software': _software_versions(),
        'files': {
            name: Path(os.path.relpath(Path(value).resolve(), repo_root)).as_posix()
            for name, value in paths.items()
        },
    }
    # Serialize before opening: invalid values must not leave an empty record.
    serialized = json.dumps(metadata, indent=2, allow_nan=False) + '\n'
    readme = _run_readme(metadata)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    with readme_path.open('x', encoding='utf-8') as stream:
        stream.write(readme)

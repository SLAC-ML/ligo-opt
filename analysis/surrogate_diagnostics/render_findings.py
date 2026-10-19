"""Render a findings Markdown file as a self-contained HTML report.

Uses the local pandoc executable and embeds linked figures and CSV/JSON evidence.
Run with the repository's ligoopt Python; no simulation or training is performed.
"""

import argparse
import base64
import csv
import html
import io
from pathlib import Path
import re
import shutil
import subprocess


def render(source, destination):
    source, destination = Path(source), Path(destination)
    pandoc = shutil.which("pandoc")
    if not pandoc:
        # pypandoc ships its executable outside PATH in some environments.
        import pypandoc
        pandoc = pypandoc.get_pandoc_path()
    body = subprocess.run(
        [pandoc, "--from=gfm", "--to=html5", str(source)],
        check=True, capture_output=True, text=True,
    ).stdout
    template = source.with_name("RUN3_FINDINGS.html").read_text()
    styles = "\n".join(re.findall(r"<style[^>]*>.*?</style>", template, re.S))
    artifacts = []

    def embed_image(match):
        path = (source.parent / html.unescape(match[1])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/{path.suffix[1:]};base64,{data}"'

    body = re.sub(r'src="([^\"]+\.(?:png|jpg|jpeg|svg))"', embed_image, body)

    def embed_artifact(match):
        href, label = html.unescape(match[1]), match[2]
        path = (source.parent / href).resolve()
        if path.suffix not in (".json", ".csv") or not path.is_file():
            return match[0]
        key = "artifact-" + re.sub(r"[^a-z0-9]+", "-", href.lower()).strip("-")
        if not any(item[0] == key for item in artifacts):
            artifacts.append((key, path, href, label))
        return f'<a href="#{key}">{label}</a>'

    body = re.sub(r'<a\s+href="([^\"]+)">(.*?)</a>', embed_artifact, body, flags=re.S)
    appendix = ['<section id="supporting-artifacts"><h2>Supporting artifacts</h2>',
                '<p>Evidence files are embedded below so this report can be shared on its own.</p>']
    for key, path, href, label in artifacts:
        text = path.read_text()
        data = base64.b64encode(path.read_bytes()).decode()
        mime = "application/json" if path.suffix == ".json" else "text/csv"
        if path.suffix == ".csv":
            rows = list(csv.reader(io.StringIO(text)))
            head = ''.join(f'<th>{html.escape(v)}</th>' for v in rows[0])
            content = '<div class="artifact-csv"><table><thead><tr>' + head + '</tr></thead><tbody>'
            content += ''.join('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in row) + '</tr>' for row in rows[1:])
            content += '</tbody></table></div>'
        else:
            content = f'<pre class="artifact-json">{html.escape(text)}</pre>'
        appendix.append(
            f'<section class="artifact" id="{key}"><h3>{label}</h3>'
            f'<p class="artifact-path"><code>{html.escape(href)}</code></p>'
            f'<p><a download="{path.parent.name}_{path.name}" href="data:{mime};base64,{data}">'
            f'Download {path.suffix[1:].upper()}</a></p>'
            f'<details><summary>View {path.suffix[1:].upper()} ({path.stat().st_size:,} bytes)</summary>'
            f'{content}</details></section>'
        )
    appendix.append('</section>')
    title = next(line[2:] for line in source.read_text().splitlines() if line.startswith('# '))
    destination.write_text(
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(title)}</title>{styles}</head><body>{body}'
        + '\n'.join(appendix) + '</body></html>\n'
    )
    print(f'{destination}: {len(artifacts)} embedded evidence files')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)

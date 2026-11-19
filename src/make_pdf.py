"""Builds Report/REPORT.pdf from Report/REPORT.md using a locally installed Edge or Chrome.

Run: python -m src.make_pdf

REPORT.md starts with a few "key: value" lines (title, subtitle, author, date) and a line with
three dashes; the rest is the report body. The PDF gets a title page, a table of contents with
page numbers, and a page number at the foot of every page after the title page.
"""
import html
import os
import re
import shutil
import subprocess

import markdown
from pypdf import PdfReader

CSS = """
@page { size: A4; margin: 24mm 22mm 22mm 22mm;
        @bottom-center { content: counter(page); font-family: Georgia, serif; font-size: 9.5pt; color: #444; } }
@page :first { @bottom-center { content: none; } }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.5; color: #111; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p { margin: 6pt 0; text-align: left; }
li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 10pt 0; }
th, td { border: 1px solid #bbb; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f2f2f0; }
tr, img, h2, h3 { page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
img { max-width: 100%; }
.cover { height: 235mm; display: flex; flex-direction: column; justify-content: center;
         text-align: center; page-break-after: always; }
.cover h1 { font-size: 24pt; line-height: 1.25; margin: 0 0 14pt; }
.cover .subtitle { font-size: 13pt; color: #333; margin-bottom: 44pt; }
.cover .author { font-size: 14pt; margin-bottom: 6pt; }
.cover .date { font-size: 11pt; color: #444; }
.toc { page-break-after: always; }
.toc h2 { margin-top: 0; }
.toc table { font-size: 11pt; }
.toc td { border: none; padding: 4pt 0; }
.toc td.page { text-align: right; width: 30pt; }
.toc tr.sub td.name { padding-left: 16pt; font-size: 10.5pt; }
.page-break { page-break-before: always; }
"""


def find_browser():
    candidates = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("chromium"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    return next(c for c in candidates if c and os.path.exists(c))


def render(page_html, target):
    html_path = os.path.abspath("Report/REPORT.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(page_html)
    subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={target}", "file:///" + html_path.replace("\\", "/")],
                   check=True, capture_output=True)
    os.remove(html_path)


def squash(text):
    return re.sub(r"\s+", "", text)


def main():
    with open("Report/REPORT.md", encoding="utf-8") as fh:
        front, body = fh.read().split("\n---\n", 1)
    meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
    body_html = markdown.markdown(body, extensions=["tables"])
    headings = re.findall(r"<(h2|h3)>(.*?)</\1>", body_html)

    def page(pages):
        rows = "".join(
            f"<tr class='{'sub' if tag == 'h3' else 'main'}'><td class='name'>{title}</td>"
            f"<td class='page'>{pages.get(title, '')}</td></tr>" for tag, title in headings)
        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                f"<title>{html.escape(meta['title'])}</title><style>{CSS}</style></head><body>"
                f"<div class='cover'><h1>{html.escape(meta['title'])}</h1>"
                f"<div class='subtitle'>{html.escape(meta['subtitle'])}</div>"
                f"<div class='author'>{html.escape(meta['author'])}</div>"
                f"<div class='date'>{html.escape(meta['date'])}</div>"
                f"<div class='date' style='margin-top:28pt'>{html.escape(meta.get('repository', ''))}</div></div>"
                f"<div class='toc'><h2>Table of contents</h2><table>{rows}</table></div>"
                f"{body_html}</body></html>")

    target = os.path.abspath("Report/REPORT.pdf")
    render(page({}), target)                      # first pass: find the page of every heading
    texts = [squash(p.extract_text() or "") for p in PdfReader(target).pages]
    pages, start = {}, 2                          # the body starts after the title page and contents
    for _, title in headings:
        key = squash(html.unescape(re.sub(r"<.*?>", "", title)))
        for number in range(start, len(texts)):
            if key in texts[number]:
                pages[title] = number + 1
                start = number
                break
    render(page(pages), target)                   # second pass: contents with page numbers
    print("wrote Report/REPORT.pdf,", len(texts), "pages,", len(pages), "of", len(headings), "headings located")


if __name__ == "__main__":
    main()

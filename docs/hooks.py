# Author: Rhyan Barrett
# Part of FieldMACE, distributed under the MIT License (see LICENSE.md)
"""mkdocs hook: build the pages from the repository's own Markdown before every build.

README.md -> index.md, CHANGELOG.md -> changelog.md, tutorials/README.md -> tutorials/index.md,
tutorials/NN_*/README.md -> tutorials/NN_*/index.md (with their figures). Links to files that are not
pages (scripts, data, ...) are pointed at GitHub, so the site has no dead links.
"""
import glob
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
GITHUB = "https://github.com/rhyan10/FieldMACE/blob/master/"
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")


def _rewrite(text, src_dir, dst_dir):
    def fix(m):
        url = m.group(2)
        if re.match(r"^[a-z]+:", url) or url.startswith("#"):
            return m.group(0)
        path, _, anchor = url.partition("#")
        target = os.path.normpath(os.path.join(src_dir, path))
        rel_to_root = os.path.relpath(target, ROOT)
        page = None
        if os.path.isdir(target) and os.path.exists(os.path.join(target, "README.md")):
            page = os.path.join(DOCS, rel_to_root, "index.md") if rel_to_root != "." else os.path.join(DOCS, "index.md")
        elif target.endswith("README.md"):
            page = os.path.join(DOCS, os.path.dirname(rel_to_root), "index.md")
        elif rel_to_root.startswith("docs" + os.sep) and rel_to_root.endswith(".md"):
            page = os.path.join(ROOT, rel_to_root)  # a hand-written docs page
        elif rel_to_root == "CHANGELOG.md":
            page = os.path.join(DOCS, "changelog.md")
        elif m.group(1).startswith("!") and os.path.exists(target):  # images are copied
            page = os.path.join(DOCS, rel_to_root)
        if page is None:
            return f"{m.group(1)}{GITHUB}{rel_to_root}{'#' + anchor if anchor else ''}{m.group(3)}"
        new = os.path.relpath(page, dst_dir)
        return f"{m.group(1)}{new}{'#' + anchor if anchor else ''}{m.group(3)}"

    return LINK.sub(fix, text)


def _copy_md(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(_rewrite(text, os.path.dirname(src), os.path.dirname(dst)))


def on_pre_build(config, **kwargs):
    _copy_md(os.path.join(ROOT, "README.md"), os.path.join(DOCS, "index.md"))
    _copy_md(os.path.join(ROOT, "CHANGELOG.md"), os.path.join(DOCS, "changelog.md"))
    _copy_md(os.path.join(ROOT, "tutorials", "README.md"), os.path.join(DOCS, "tutorials", "index.md"))
    for d in sorted(glob.glob(os.path.join(ROOT, "tutorials", "[0-9][0-9]_*"))):
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "README.md")):
            continue
        _copy_md(os.path.join(d, "README.md"), os.path.join(DOCS, "tutorials", name, "index.md"))
        for sub in ("figures", "results"):
            for img in glob.glob(os.path.join(d, sub, "*.png")):
                out = os.path.join(DOCS, "tutorials", name, sub, os.path.basename(img))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy(img, out)

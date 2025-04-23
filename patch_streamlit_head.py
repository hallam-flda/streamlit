import pathlib, shutil, sys
import streamlit

static_dir = pathlib.Path(streamlit.__file__).parent / "static"
index_file = static_dir / "index.html"
backup_file = static_dir / "index.html.bak"

if not backup_file.exists():
    shutil.copy(index_file, backup_file)

orig = index_file.read_text(encoding="utf-8")
snippet = pathlib.Path("/app/head.html").read_text(encoding="utf-8")

if "<head>" in orig and snippet not in orig:
    new = orig.replace("<head>", "<head>\n" + snippet)
    index_file.write_text(new, encoding="utf-8")
else:
    sys.stderr.write("Couldn’t patch index.html – check paths.\n")
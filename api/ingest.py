import os, glob, requests


API_URL = os.environ.get("API_URL", "http://localhost:8000")
DOCS_DIR = os.environ.get("DOCS_DIR", "./data/docs")
NAMESPACE = os.environ.get("NAMESPACE", "default")


files = []
for ext in ("*.txt", "*.md", "*.pdf"):
files.extend(glob.glob(os.path.join(DOCS_DIR, ext)))


if not files:
print("Aucun fichier à ingérer dans", DOCS_DIR)
raise SystemExit(0)


mp = []
for p in files:
mp.append(("files", (os.path.basename(p), open(p, "rb"))))


resp = requests.post(f"{API_URL}/ingest", files=mp, data={"namespace": NAMESPACE})
print(resp.status_code, resp.text)
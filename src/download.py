"""Part 1: download the raw file once. Never overwrite it."""
import hashlib
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Official location: the data file linked from the JSE article page (De Cock, 2011).
DATA_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"
# Public mirror of the same file, used only if the JSE server cannot be reached
# (rasbt/machine-learning-book, ch09). EXPECTED_MD5 is the MD5 of the copy used for this
# submission; a mismatch only prints a warning, the shape check is the hard gate.
MIRROR_URL = "https://raw.githubusercontent.com/rasbt/machine-learning-book/main/ch09/AmesHousing.txt"
EXPECTED_MD5 = "c2997eb69ab4bf946b9fbdb7e6805e36"
EXPECTED_SHAPE = (2930, 82)


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "DS5110 HW 5 (educational use)"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def main(dest: str) -> None:
    path = Path(dest)
    if path.exists():
        print(f"{path} exists; raw data is never overwritten.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    data, used = None, None
    for url in (DATA_URL, MIRROR_URL):
        try:
            data, used = fetch(url), url
            break
        except (URLError, TimeoutError, OSError) as e:
            print(f"Could not download {url}: {e}")
    if data is None:
        sys.exit("Download failed from both URLs. Place AmesHousing.txt in data/raw/ by hand.")

    md5 = hashlib.md5(data).hexdigest()
    # Shape check on arrival: header line + one line per row, tab-separated.
    lines = data.decode("latin-1").splitlines()
    shape = (len(lines) - 1, len(lines[0].split("\t")))
    if shape != EXPECTED_SHAPE:
        sys.exit(f"Unexpected shape {shape}, expected {EXPECTED_SHAPE}. File not saved.")
    if md5 != EXPECTED_MD5:
        print(f"WARNING: MD5 {md5} differs from the recorded {EXPECTED_MD5}.")

    # "x" mode refuses to overwrite even if the file appeared in the meantime.
    with open(path, "xb") as f:
        f.write(data)
    print(f"Downloaded {used} -> {path}  rows={shape[0]} cols={shape[1]} md5={md5}")


if __name__ == "__main__":
    main(sys.argv[1])

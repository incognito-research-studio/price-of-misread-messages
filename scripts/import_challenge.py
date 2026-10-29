"""Import the frozen WP2 challenge texts into stream files (docs/research/challenge/README.md).

usage: python scripts/import_challenge.py PILOT_STREAMS_DIR REWRITES_YAML OUT_DIR MANIFEST_JSON

Takes the challenge streams named in REWRITES_YAML from PILOT_STREAMS_DIR (the template pilot),
swaps in every verified rewrite and re-anchors each gold delta's span to its evidence. Nothing is
written unless every entry passes (cdd.benchmark.challenge.import_challenge); the problems are
listed instead. OUT_DIR gets the stream files only (anything else there would be read as a
stream); MANIFEST_JSON gets their hashes and the rewrites file's hash, for the freeze record.
The span parser is never run here.
"""

import hashlib
import json
import sys
from pathlib import Path

from cdd.benchmark.challenge import ChallengeError, import_challenge, load_rewrites
from cdd.benchmark.stream import load_streams


def main() -> None:
    pilot, rewrites_path, out, manifest_path = (
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
    )
    rewrites, sha = load_rewrites(rewrites_path)
    wanted = {r.id.rsplit("-m", 1)[0] for r in rewrites}
    streams = [s for s in load_streams(pilot) if s.id in wanted]
    missing = sorted(wanted - {s.id for s in streams})
    if missing:
        sys.exit(f"streams not found in {pilot}: {missing}")
    try:
        imported = import_challenge(streams, rewrites, sha)
    except ChallengeError as exc:
        print(f"{len(exc.problems)} problems; nothing written:")
        print("\n".join(f"   {p}" for p in exc.problems))
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"rewrites_sha256": sha, "streams": []}
    for stream in imported:
        path = out / f"{stream.id}.json"
        stream.save(path)
        manifest["streams"].append(
            {
                "id": stream.id,
                "messages": len(stream.messages),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    messages = sum(len(s.messages) for s in imported)
    print(f"imported {len(imported)} streams, {messages} messages, into {out}")


if __name__ == "__main__":
    main()

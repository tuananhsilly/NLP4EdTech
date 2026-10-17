#!/usr/bin/env bash
# Fetch the raw KB sources listed in data/source_manifest.yaml into data/sources/ and verify
# their SHA-256 checksums. The manifest is the single source of truth: repo, commit, module
# ids and checksums are all read from it.
#
# Usage: scripts/fetch_sources.sh            (run from anywhere; needs git >= 2.27)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/data/source_manifest.yaml"
DEST="$ROOT/data/sources/osbooks-prealgebra-bundle"

field() { grep -m1 -E "^\s+$1:" "$MANIFEST" | sed -E "s/^\s+$1:\s*([^ #]+).*/\1/"; }
REPO_URL="$(field repo_url)"
COMMIT="$(field commit)"
MODULES=$(grep -oE '\bid: m[0-9]+' "$MANIFEST" | awk '{print "modules/" $2}')

if [[ -e "$DEST" ]]; then
  echo "error: $DEST already exists; delete it first to re-fetch" >&2
  exit 1
fi

# Partial clone: only the files selected by sparse checkout are downloaded (no images).
git clone --quiet --no-checkout --filter=blob:none --sparse "$REPO_URL" "$DEST"
git -C "$DEST" fetch --quiet --depth 1 origin "$COMMIT"
# shellcheck disable=SC2086  # MODULES is a whitespace-separated list on purpose
git -C "$DEST" sparse-checkout set --no-cone /LICENSE /README.md collections/ META-INF/ $MODULES
git -C "$DEST" checkout --quiet "$COMMIT"

# Build "<sha256>  <path>" lines from the manifest: top-level files, then each module's
# sha256 paired with the most recent module id seen above it.
checks=$(awk '
  /^ *(LICENSE|collections\/[^:]+):/ { gsub(":", "", $1); print $2 "  " $1 }
  /id: m[0-9]+/ { match($0, /m[0-9]+/); mod = substr($0, RSTART, RLENGTH) }
  /sha256:/     { match($0, /[0-9a-f]{64}/); print substr($0, RSTART, RLENGTH) "  modules/" mod "/index.cnxml" }
' "$MANIFEST")

echo "fetched $REPO_URL @ ${COMMIT:0:12} into ${DEST#"$ROOT"/}"
(cd "$DEST" && sha256sum --check --strict <<<"$checks")

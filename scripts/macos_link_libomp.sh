#!/bin/bash
# LightGBM's macOS wheel expects Homebrew's libomp. If it is missing, point
# LightGBM at the libomp copy that ships inside scikit-learn's wheel instead.
# Only files inside the virtualenv are touched. `brew install libomp` makes this unnecessary.
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || exit 0
PY="${1:-.venv/bin/python}"
if "$PY" -c "import lightgbm" 2>/dev/null; then echo "lightgbm loads fine; nothing to do"; exit 0; fi
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/lightgbm/lib/lib_lightgbm.dylib"
[[ -f "$SITE/sklearn/.dylibs/libomp.dylib" ]] || { echo "sklearn libomp not found; run: brew install libomp"; exit 1; }
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB"
codesign --force --sign - "$LIB"
"$PY" -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'now loads using sklearn libomp')"

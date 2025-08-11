set -e
conda run --no-capture-output -n env python -m pip freeze
conda run --no-capture-output -n env python server.py
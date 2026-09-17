# Environment

We use `uv` as our package manager. To install it look on the [installation page](https://docs.astral.sh/uv/getting-started/installation/). For windows I would probably just recommend installing it via pip. If you go the method of curl-ing (or however they do it on windows) and running the installer script via a shell, you do have the ability to let ai look over the installer script.

`uv` doesn't require sudo/admin privileges, so can also be easily installed on snellius via the `curl ... | sh`

## Running code

After that it is just `uv run python src/main.py --dataset TOY2` or `uv run make data/SEGTHOR`

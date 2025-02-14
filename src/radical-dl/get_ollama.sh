#!/bin/sh

# download the Ollama binary from the official website and installs it in the
# current directory.

TGT=$1
test -z $1 && TGT="."
mkdir -p $TGT
tgt=$TGT/ollama

ARCH=$(uname -m)
case "$ARCH" in
    x86_64       ) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *            )
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

if test -f $tgt.md5
then
    md5=$(md5sum $tgt)
    if test "$md5" = "$(cat $tgt.md5)"
    then
        echo "Ollama is already installed."
        exit 0
    fi
fi

test -f $tgt && cont='-C -' || cont=''

curl --fail $cont --show-error --location --progress-bar -o $tgt \
    "https://ollama.com/download/ollama-linux-${ARCH}"

chmod +x $tgt
V=$($tgt --version)

if ! test -z $V; then
    md5sum $tgt > $tgt.md5
    echo "Ollama $V has been installed successfully."
    echo "for available models, check https://ollama.com/library"
    echo "run `$tgt start` to start the server (in a separate shell)."
    echo "run `$tgt pull <model>` to download a model."
    echo "run `$tgt run <model>` to run a model."

    exit 1
else
    echo "Failed to install Ollama."
    exit 1
fi

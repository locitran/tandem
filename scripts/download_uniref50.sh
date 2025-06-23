#!/bin/bash
# Adapted to use wget instead of aria2c
# Original from AlphaFold2 scripts
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v wget &> /dev/null ; then
    echo "Error: wget could not be found. Please install wget (sudo apt install wget)."
    exit 1
fi

DOWNLOAD_DIR="$1"
SOURCE_URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
BASENAME=$(basename "${SOURCE_URL}")

# Start timer
SECONDS=0

mkdir -p "${DOWNLOAD_DIR}"
wget -P "${DOWNLOAD_DIR}" "${SOURCE_URL}"

pushd "${DOWNLOAD_DIR}"
gunzip "${BASENAME}"
popd

# Display elapsed time
duration=$SECONDS
echo "Download and extraction completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."
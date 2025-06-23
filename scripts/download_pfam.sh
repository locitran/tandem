#!/bin/bash
# This script downloads the Pfam database from the EBI FTP server.
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

# Check if aria2c is installed
if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

# Check if hmmer is installed
if ! command -v hmmpress &> /dev/null ; then
    echo "Error: hmmer could not be found. Please install hmmer (sudo apt install hmmer)."
    exit 1
fi

DOWNLOAD_DIR="$1"
SOURCE_URL="http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz"
BASENAME=$(basename "${SOURCE_URL}")
SOURCE_URL2="http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
BASENAME2=$(basename "${SOURCE_URL2}")
# Start timer
SECONDS=0

mkdir -p "${DOWNLOAD_DIR}"
aria2c "${SOURCE_URL}" --dir="${DOWNLOAD_DIR}"
aria2c "${SOURCE_URL2}" --dir="${DOWNLOAD_DIR}"
pushd "${DOWNLOAD_DIR}"
gunzip "${BASENAME}"
gunzip "${BASENAME2}"
# Prepare Pfam database for HMMER by creating binary files, (without the .gz extension)
BASENAME2="${BASENAME2%.gz}"
hmmpress "${BASENAME2}"
popd

# Display elapsed time
duration=$SECONDS
echo "Download and extraction completed in $(($duration / 60)) minutes and $(($duration % 60)) seconds."

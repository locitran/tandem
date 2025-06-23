# Docker

**Step 1**: Assume you are in `test` folder :  `./test/`
clone the folder
```bash
mkdir tandem
cd tandem
git clone https://github.com/locitran/tandem-dimple.git .
```

Now we should be in the path/to/tandem folder.
`ls` command should show `src/` `data/` `models/` etc.

**Step 2**: Build docker image using original Dockerfile (please not put databases inside docker images)
```bash
docker build -t tandem -f docker/Dockerfile .
```

**Step 3**: Download 2 databases: 
```bash
bash scripts/download_pfam.sh data/pfamdb # 1.5G, ~1.5m
bash scripts/download_consurf_db.sh data/consurf/db # 2.5G, ~2m
# Please skip this database for now
# We will download this database later
bash scripts/download_uniref90.sh data/consurf # 90G, ~127m
# Consider downloading uniref50 instead of uniref90 due to the size
bash scripts/download_uniref50.sh data/consurf # 90G, ~127m
```
After this step, we will have `path/to/tandem//data/pfamdb` folder and `path/to/tandem/data/consurf/db/2024-10-08` folder. Make sure you have the correct path to these databases.

**Step 4**: Run image/container
```bash
# cwd: path/to/tandem
# pfamdb: path/to/tandem/data/pfamdb
# consurf: path/to/tandem/data/consurf
docker run -it \
  -v .:/tandem \
  -w /tandem \
  tandem:latest bash \
  -c "source activate tandem && python test/input_as_list_SAVs.py"
```

# Build from Scratch, without Docker

**Step 1**: Assume you are in `test` folder :  `./test/`
clone the folder
```bash
mkdir tandem
cd tandem
git clone https://github.com/locitran/tandem-dimple.git .
```

**Step 2**: Install dependencies
We assume you have already installed Anaconda or Miniconda. If not, please install it first.
```bash
conda create -n tandem python=3.11.11
conda activate tandem
pip install -r requirements.txt
sudo apt install hmmer
```

**Step 3**: Download 2 databases: 
```bash
bash scripts/download_pfam.sh data/pfamdb # 1.5G, ~1.5m
bash scripts/download_consurf_db.sh data/consurf/db # 2.5G, ~2m
# Please skip this database for now
# We will download this database later
bash scripts/download_uniref90.sh data/consurf # 90G, ~127m
# dependencies for ConSurf tool
conda install -c conda-forge -c bioconda mmseqs2
sudo apt install cd-hit
sudo apt install prottest 
sudo apt install ncbi-blast+
sudo apt install mafft
sudo apt install clustalw
sudo apt install muscle
```
**Step 4**: Test the installation
```bash
python test/input_as_list_SAVs.py
```

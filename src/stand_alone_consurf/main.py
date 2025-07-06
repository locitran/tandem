from .consurf import *
from . import GENERAL_CONSTANTS

import os, re
from datetime import datetime
from datetime import date
import time
from ..utils.logger import LOGGER
basedir = os.path.dirname(os.path.abspath(__file__))

def run(
        msa=None,
        query="",
        tree=None,
        structure=None,
        chain=None,
        iterations="1",
        cutoff="0.0001",
        DB=None,
        MAX_HOMOLOGS="150",
        closest=False,
        MAX_ID="95",
        MIN_ID="35",
        align="MAFFT",
        Maximum_Likelihood=False,
        model="BEST",
        seq=None,
        Nuc=False,
        algorithm="HMMER",
        work_dir='.',
        cif=False
    ):
    cwd = os.getcwd()
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Set up
    form = {}
    vars = {}
    form['msa_SEQNAME'] = query
    form['pdb_FILE'] = structure
    vars['user_msa_file_name'] = msa
    form['tree_name'] = tree
    form['ITERATIONS'] = iterations
    vars['uploaded_Seq'] = seq
    vars['working_dir'] = work_dir if work_dir.endswith("/") else work_dir + "/"
    vars['protein_db'] = DB
    form['E_VALUE'] = cutoff
    form['Homolog_search_algorithm'] = algorithm.upper()
    form['MAX_NUM_HOMOL'] = MAX_HOMOLOGS
    form['MAX_REDUNDANCY'] = MAX_ID
    form['MIN_IDENTITY'] = MIN_ID
    form['MSAprogram'] = align.upper()
    form['PDB_chain'] = chain
    form['SUB_MATRIX'] = model
    # Validation checks
    if form['pdb_FILE'] is None and vars['user_msa_file_name'] is None and vars['uploaded_Seq'] is None:
        raise ValueError("You must provide at least one of: structure (PDB), MSA, or sequence")
    if not form['ITERATIONS'].isdigit():
        raise ValueError("--iterations should be a positive whole number.")
    if not form['MAX_NUM_HOMOL'].isdigit():
        raise ValueError("--MAX_HOMOLOGS should be a positive whole number.")
    if not form['MIN_IDENTITY'].isdigit():
        raise ValueError("--MIN_ID should be a positive whole number.")
    if not form['MAX_REDUNDANCY'].isdigit():
        raise ValueError("--MAX_ID should be a positive whole number.")
    if form['MSAprogram'] not in ["MAFFT", "PRANK", "MUSCLE", "CLUSTALW"]:
        raise ValueError("--align should be MAFFT, PRANK, MUSCLE or CLUSTALW.")
    if form['SUB_MATRIX'] not in ["JTT", "LG", "mtREV", "cpREV", "WAG", "Dayhoff", "BEST"]:
        raise ValueError("--model should be JTT, LG, mtREV, cpREV, WAG or Dayhoff.")
    if form['Homolog_search_algorithm'] not in ["BLAST", "HMMER", "MMSEQS2"]:
        raise ValueError("--algorithm should be HMMER, BLAST or MMseqs2.")
    try:
        float(form['E_VALUE'])
    except ValueError:
        raise ValueError("--cutoff should be a positive number.")
    form['DNA_AA'] = "Nuc" if Nuc else "AA"
    vars['cif_or_pdb'] = "cif" if cif else "pdb"
    form['best_uniform_sequences'] = "best" if closest else "uniform"
    form['ALGORITHM'] = "LikelihoodML" if Maximum_Likelihood else "Bayes"
    
    vars['tree_file'] = os.path.join(vars['working_dir'], "TheTree.txt")
    vars['msa_fasta'] = os.path.join(vars['working_dir'], "msa_fasta.aln")

    vars['running_mode'] = determine_mode(form, vars)
    vars['BLAST_out_file'] = os.path.join(vars['working_dir'], "blast_out.txt")

    if form['DNA_AA'] == "AA": # proteins
        vars['protein_or_nucleotide'] = "proteins"
        vars['Msa_percentageFILE'] = os.path.join(vars['working_dir'], "msa_aa_variety_percentage.csv")
    else: # nucleotides
        vars['protein_or_nucleotide'] = "nucleotides"
        vars['Msa_percentageFILE'] = os.path.join(vars['working_dir'], "msa_nucleic_acids_variety_percentage.csv")

    vars['All_Outputs_Zip'] = os.path.join(vars['working_dir'], "Consurf_Outputs.zip")
    if form['pdb_FILE'] is not None: # User PDB
        match = re.search(r'([^\/]+)$', form['pdb_FILE'])
        if match:
            vars['Used_PDB_Name'] = match.group(1)
        vars['Used_PDB_Name'] = re.sub(r"[() ]", r"_", vars['Used_PDB_Name'])
        match = re.search(r'(\S+)\.', vars['Used_PDB_Name'])
        if match:
            vars['Used_PDB_Name'] = match.group(1)
    else: # ConSeq Mode, No Model
        vars['Used_PDB_Name'] = "no_model"

    if form['E_VALUE'].isdigit() and form['E_VALUE'] != "0":
        # if the user inserted an integer, we turn it to a fraction
        number_of_zeros = int(form['E_VALUE'])
        form['E_VALUE'] = "0."
        i = 1
        while i < number_of_zeros:
            form['E_VALUE'] += "0"
            i += 1
        form['E_VALUE'] += "1"

    if int(form['MAX_REDUNDANCY']) >= 100:
        vars['hit_redundancy'] = 99.999999
    else:
        vars['hit_redundancy'] = float(form['MAX_REDUNDANCY'])

    form['Run_Number'] = "0"
    vars['hit_min_length'] = GENERAL_CONSTANTS.FRAGMENT_MINIMUM_LENGTH # minimum length of homologs
    vars['min_num_of_hits'] = GENERAL_CONSTANTS.MINIMUM_FRAGMENTS_FOR_MSA # minimum number of homologs
    vars['FINAL_sequences'] = os.path.join(vars['working_dir'], "query_final_homolougs.fasta") # finial homologs for creating the MSA
    vars['FINAL_sequences_html'] = os.path.join(vars['working_dir'], "query_final_homolougs.html") # html files showing the finial homologs to the user
    vars['submission_time'] = str(datetime.now())
    vars['date'] = date.today().strftime("%d/%m/%Y")
    vars['time_table'] = []
    vars['current_time'] = time.time()
    vars['gradesPE'] = vars['Used_PDB_Name'] + "_consurf_grades.txt" # file with consurf output
    vars['gradesPE'] = os.path.join(vars['working_dir'], vars['gradesPE'])
    vars['zip_list'] = []

    # pymol and chimera scripts
    vars['chimera_color_script'] = os.path.join(basedir, "color_consurf_chimerax_session.py")
    vars['chimera_color_script_CBS'] = os.path.join(basedir, "color_consurf_CBS_chimerax_session.py")
    vars['pymol_color_script_isd'] = os.path.join(basedir, "color_consurf_pymol_isd_session.py")
    vars['pymol_color_script_CBS_isd'] = os.path.join(basedir, "color_consurf_CBS_pymol_isd_session.py")

    vars['msa_clustal'] = os.path.join(vars['working_dir'], "msa_clustal.aln") # if the file is not in clustal format, we create a clustal copy of it

    vars['Colored_Seq_PDF'] = os.path.join(vars['working_dir'], "consurf_colored_seq.pdf")
    vars['Colored_Seq_CBS_PDF'] = os.path.join(vars['working_dir'], "consurf_colored_seq_CBS.pdf")

    vars['gradesPE_Output'] = [] # an array to hold all the information that should be printed to gradesPE
    # in each array's cell there is a hash for each line from r4s.res.
    # POS: position of that aa in the sequence ; SEQ : aa in one letter ;
    # GRADE : the given grade from r4s output ; COLOR : grade according to consurf's scale

    vars['zip_list'].append(vars['tree_file'])
    vars['zip_list'].append(vars['gradesPE'])
    vars['zip_list'].append(vars['Msa_percentageFILE'])
    vars['zip_list'].append(vars['Colored_Seq_PDF'])
    vars['zip_list'].append(vars['Colored_Seq_CBS_PDF'])
    vars['zip_list'].append(vars['msa_fasta'])

    ## mode : include pdb

    # create a pdbParser, to get various info from the pdb file
    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":
        extract_data_from_model(form, vars)

    """
    ## mode : only protein sequence

    # if there is only protein sequence: we upload it.
    elif vars['running_mode'] == "_mode_no_pdb_no_msa":

        upload_protein_sequence()
    """
    ## mode : no msa - with PDB or without PDB
    LOGGER.timeit("_ConSurf_tool")
    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_no_pdb_no_msa":
        no_MSA(form, vars)
    ## mode : include msa
    elif vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_msa" or vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":
        extract_data_from_MSA(form, vars)

    if form['SUB_MATRIX'] == "BEST":
        #vars['best_fit'] = True
        find_best_substitution_model(form, vars)
    else:
        vars['best_fit'] = "model_chosen"

    run_rate4site_old(form, vars)
    assign_colors_according_to_r4s_layers(form, vars)
    write_MSA_percentage_file(form, vars)

    # mode : include pdb

    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":
        consurf_create_output(form, vars)

    # ## mode : ConSeq - NO PDB

    if vars['running_mode'] == "_mode_msa" or vars['running_mode'] == "_mode_no_pdb_no_msa" or vars['running_mode'] == "_mode_msa_tree":
        conseq_create_output(form, vars)

    # zip_all_outputs(vars)
    LOGGER.report('ConSurf tool computed in %.2fs.', '_ConSurf_tool')

    os.chdir(cwd)
    return vars['gradesPE']
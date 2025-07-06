from . import GENERAL_CONSTANTS

import re
import os
import shutil
import subprocess
import fpdf 
import math

from datetime import datetime
from zipfile import ZipFile
from Bio import AlignIO
from Bio import SearchIO
from Bio import Align
from Bio.Align import substitution_matrices
from ..utils.logger import LOGGER


basedir = os.path.dirname(os.path.abspath(__file__))

bayesInterval = 3
ColorScale = {0 : 9, 1 : 8, 2 : 7, 3 : 6, 4 : 5, 5 : 4, 6 : 3, 7 : 2, 8 : 1}

def get_form_direct(
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
    work_dir=None,
    cif=False
):
    if work_dir is None:
        raise ValueError("Missing required argument: work_dir")
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

    vars['run_log'] = "log.txt"
    vars['tree_file'] = "TheTree.txt"
    vars['msa_fasta'] = "msa_fasta.aln" 

    return form, vars

def determine_mode(form, vars):

    mode = ""

    # determine the running mode and the pdb file mode
    #_mode_pdb_no_msa : (pdb_ID OR pdb_FILE) AND _form_pdb_chain AND build MSA parameters
    #_mode_pdb_msa : (pdb_ID OR pdb_FILE) AND _form_pdb_chain AND msa_FILE AND msa_SEQ_name
    #_mode_pdb_msa_tree: (pdb_ID OR pdb_FILE) AND _form_pdb_chain AND msa_FILE AND msa_SEQ_name AND tree_FILE
    #_mode_msa : msa_FILE AND msa_SEQ_name
    #_mode_msa_tree : msa_FILE AND msa_SEQ_name AND tree_FILE
    #_mode_no_pdb_no_msa : protein_SEQ or uploaded_SEQ AND build MSA parameters
    
    vars['protein_seq'] = os.path.join(vars['working_dir'], "protein_seq.fas") # a fasta file with the protein sequence from PDB or from protein seq input
    if form['pdb_FILE'] is not None:
        vars['pdb_file_name'] = os.path.join(vars['working_dir'], "pdb_file.ent")
        if not (os.path.isfile(form['pdb_FILE']) and os.access(form['pdb_FILE'], os.R_OK)):
            LOGGER.error("The PDB file needs to be accessible.")
            
        shutil.copy(form['pdb_FILE'], vars['pdb_file_name'])
        if vars['user_msa_file_name'] is not None:
            if not (os.path.isfile(vars['user_msa_file_name']) and os.access(vars['user_msa_file_name'], os.R_OK)):
                LOGGER.error("The MSA file needs to be accessible.")
                
            MSA_sequences = [] # sequence names
            get_info_from_msa(MSA_sequences, form, vars)
            if form['tree_name'] is not None:
                if not (os.path.isfile(form['tree_name']) and os.access(form['tree_name'], os.R_OK)):
                    LOGGER.error("The tree file needs to be accessible.")
                check_msa_tree_match(MSA_sequences, vars)
                mode = "_mode_pdb_msa_tree"
            else:
                mode = "_mode_pdb_msa"
        else:
            if vars['protein_db'] is None: #or not (os.path.isfile(vars['protein_db']) and os.access(vars['protein_db'], os.R_OK)):
                LOGGER.error("Please provide a homolog database.")
            mode = "_mode_pdb_no_msa"
    else:
        if vars['user_msa_file_name'] is not None:
            if not (os.path.isfile(vars['user_msa_file_name']) and os.access(vars['user_msa_file_name'], os.R_OK)):
                LOGGER.error( "The MSA file needs to be accessible.")
            MSA_sequences = [] # sequence names
            get_info_from_msa(MSA_sequences, form, vars)
            
            if form['tree_name'] is not None:
                if not (os.path.isfile(form['tree_name']) and os.access(form['tree_name'], os.R_OK)):
                    LOGGER.error( "The tree file needs to be accessible.")
                check_msa_tree_match(MSA_sequences, vars)
                mode = "_mode_msa_tree"
            else:
                mode = "_mode_msa"
        else:
            if vars['protein_db'] is None: #or not (os.path.isfile(vars['protein_db']) and os.access(vars['protein_db'], os.R_OK)):
                LOGGER.error( "Please provide a homolog database.")
            upload_protein_sequence(form, vars)
            mode = "_mode_no_pdb_no_msa"
    return mode

def get_info_from_msa(seq_names, form, vars):
    
    # returns the name of the query sequence and fills the input array with the names of the sequences
    query_seq = ""
    msa_format = check_msa_format(vars['user_msa_file_name'])
    try:
        alignment = AlignIO.read(vars['user_msa_file_name'], msa_format)
    except:
        LOGGER.error("get_info_from_msa : can't open the file " + vars['user_msa_file_name'] + " for reading.")
    try:
        msa_fasta = os.path.join(vars['working_dir'], vars['msa_fasta'])
        MSA = open(msa_fasta, 'w')
    except:
        LOGGER.error("get_info_from_msa : can't open the file " + vars['msa_fasta'] + " for writing.")
        
    for record in alignment:
        new_seq_name = str(record.id)
        if new_seq_name in seq_names:
            LOGGER.error("The sequence %s appears more than once in the MSA." %new_seq_name)
            
        seq_names.append(new_seq_name)
        seq = str(record.seq)
        if form['msa_SEQNAME'] == new_seq_name:
            query_seq = seq
        if form['DNA_AA'] == "Nuc" and ('u' in seq or 'U' in seq):
            seq = seq.replace('u', 't')
            seq = seq.replace('U', 'T')
            LOGGER.info("Warnning: The seqeunce '" + new_seq_name + "' contains a 'U' replaced by 'T'")
        MSA.write(">%s\n%s\n" %(new_seq_name, seq))
    MSA.close()
       
    num_of_seq = len(seq_names)
    vars['unique_seqs'] = num_of_seq
    vars['final_number_of_homologoues'] = num_of_seq
    LOGGER.info("MSA contains " + str(num_of_seq) + " sequences\n")
    if num_of_seq < 5:
        LOGGER.error("The MSA file contains only " + str(num_of_seq) + " sequences. The minimal number of homologues required for the calculation is 5.")
 
    query_seq = query_seq.replace("-", "")
    query_seq = query_seq.upper()
    if query_seq == "":
        LOGGER.error( "The query sequence is not in the msa. Please choose the name of the query sequence by adding the flag --query")
        
    vars['msa_SEQNAME'] = form['msa_SEQNAME']
    #vars['query_string'] = form['msa_SEQNAME']
    vars['MSA_query_seq'] = query_seq
    """
    # there is no input seq, use msa seq instead
    vars['protein_seq_string'] = vars['MSA_query_seq']
    try:

        QUERY_FROM_MSA = open(vars['working_dir'] + vars['protein_seq'], 'w')

    except:

        LOGGER.error("get_info_from_msa : Could not open %s for writing." %vars['protein_seq'])

    QUERY_FROM_MSA.write(">" + form['msa_SEQNAME'] + "\n")
    QUERY_FROM_MSA.write(vars['MSA_query_seq'] + "\n")
    QUERY_FROM_MSA.close()
    """

def check_msa_tree_match(ref_msa_seqs, vars):

    ref_tree_nodes = []
    check_validity_tree_file(ref_tree_nodes)
    LOGGER.info("check_msa_tree_match : check if all the nodes in the tree are also in the MSA\n")

    for node in ref_tree_nodes:

        if not node in ref_msa_seqs:

            LOGGER.error("The uploaded tree file is inconsistant with the uploaded MSA file. The node '" + node + "' is found in the tree file, but there is no sequence in the MSA file with that exact name. Note that the search is case-sensitive!")

    LOGGER.info("check_msa_tree_match : check if all the sequences in the MSA are also in the tree\n")

    for seq_name in ref_msa_seqs: #check that all the msa nodes are in the tree

        if not seq_name in ref_tree_nodes:

            LOGGER.error("The uploaded MSA file is inconsistant with the uploaded tree file. The Sequence name '" + seq_name + "' is found in the MSA file, but there is no node with that exact name in the tree file. Note that the search is case-sensitive!")

    vars['unique_seqs'] = len(ref_msa_seqs)
    LOGGER.info("There are " + str(vars['unique_seqs']) + " in the MSA.\n")

def check_validity_tree_file(nodes, LOG, form, vars):
    
	# checks validity of tree file and returns an array with the names of the nodes
    try:
        TREEFILE = open(form['tree_name'], 'r')
    except:
        LOGGER.error("check_validity_tree_file : can't open the file " + form['tree_name'] + " for reading.")

    tree = TREEFILE.read()
    TREEFILE.close()
    tree.replace("\n", "")
    if tree[-1] != ';':
	    tree += ';'
    try:
        TREEFILE = open(vars['working_dir'] + vars['tree_file'], 'w')
    except:
        LOGGER.error("check_validity_tree_file : can't open the file " + vars['working_dir'] + vars['tree_file'] + " for writing.")
        
    TREEFILE.write(tree)
    TREEFILE.close()

    leftBrackets = 0
    rightBrackets = 0
    noRegularFormatChar = ""
    #nodes = []
    node_name = ""
    in_node_name = False
    in_node_score = False
    for char in tree:
        
        if char == ':':
            
            if in_node_name:
                
                nodes.append(node_name)
                
            node_name = ""
            in_node_name = False
            in_node_score = True

        elif char == '(':

            leftBrackets += 1                
            
        elif char == ')':

            rightBrackets += 1
            in_node_score = False
            
        elif char == ',':
            
            in_node_score = False
            
        elif char != ';':

            if char in "!@#$^&*~`{}'?<>" and not char in noRegularFormatChar: 

                noRegularFormatChar += " '" + char + "', "
                
            if not in_node_score:
                
                node_name += char
                in_node_name = True

    if leftBrackets != rightBrackets:

        msg = "The uploaded tree file, which appears to be in Newick format, is missing parentheses."
        LOGGER.error(msg)

    if noRegularFormatChar != "":

        msg = "The uploaded tree file, which appears to be in Newick format, ontains the following non-standard characters: " + noRegularFormatChar[:-2]
        LOGGER.error(msg)
        
    LOGGER.info("check_validity_tree_file : tree is valid\n")

    #return nodes

def check_msa_format(msa):

    try:
        
        MSA = open(msa, 'r')
        
    except:
        
        LOGGER.error("check_msa_format : Can't open %s for reading." %msa)
        
    line = MSA.readline()
    while line != "":

        line = line.strip()
        if line == "":

            line = MSA.readline()
            continue

        if line[:4] == "MSF:":

            format = "msf"
            break

        elif line[0] == '>':

            format = "fasta"
            break

        elif line[0] == '#':

            format = "nexus"
            break

        elif line[0] == 'C':

            format = "clustal"
            break

        elif line[0] == 'P':

            format = "gcg"
            break

        else:

            MSA.close()
            LOGGER.error("Unknown MSA format.")

        line = MSA.readline()

    MSA.close()
    return format

def upload_protein_sequence(form, vars):


    if os.path.exists(vars['uploaded_Seq']) and os.path.getsize(vars['uploaded_Seq']) != 0: # file fasta uploaded
        try:
            UPLOADED = open(vars['uploaded_Seq'], 'r')
        except:
            LOGGER.error("upload_protein_sequence : Cannot open the file " + vars['protein_seq'] + "for writing!")

        protein_seq_string = UPLOADED.read()
        UPLOADED.close()
        if protein_seq_string.count('>') > 1:
            LOGGER.error("The protein input <a href = \"%s\">sequence</a> contains more than one FASTA sequence. If you wish to upload MSA, please upload it as a file." %protein_seq_string)
        # delete sequence name and white spaces
        protein_seq_string = re.sub(r'>.*\n', "", protein_seq_string)
        protein_seq_string = re.sub(r'\s', "", protein_seq_string)
    else:
        LOGGER.error('upload_protein_sequence : no user sequence.')

    # we write the sequence to a file for the homologues search
    try:
        UPLOADED = open(vars['working_dir'] + vars['protein_seq'], 'w')
    except:			
        LOGGER.error("upload_protein_sequence : Cannot open the file " + vars['protein_seq'] + "for writing!")

    UPLOADED.write(">Input_seq\n" + protein_seq_string)
    UPLOADED.close()

    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
    nucleic_acids = ["A", "C", "G", "T", "U", "N"]
    if form['DNA_AA'] == "AA":
        amino = False
        for char in protein_seq_string:
            if not char.upper() in amino_acids:
                LOGGER.error("The input sequence contains the illegal character %s." %char)
            elif not char.upper() in nucleic_acids:
                amino = True
        if not amino:
            LOGGER.error("It seems that the protein input is only composed of Nucleotides (i.e. :A,T,C,G). Please note that you chose to run the server based on amino acids sequnce and not DNA / RNA sequence.<br />You may translate your sequence to amino acids and resubmit your query, or alternatively choose to analyze nucleotides.<br />")
    else:
        for char in protein_seq_string:
            if not char.upper() in amino_acids:
                LOGGER.error("The input sequence contains the illegal character %s." %char)
            elif not char.upper() in nucleic_acids:
                LOGGER.error("It seems that the input sequence contains the Amino Acid %s. Please note that you chose to run the server based on nucleotides sequnce and not protein sequence.<br />You may resubmit your query and choose to analyze Amino Acids.<br />" %char)

    vars['protein_seq_string'] = protein_seq_string
    vars['query_string'] = "Input_seq" # name of the sequence is saved for rate4site

# vars['running_mode'] = determine_mode()

def extract_data_from_model(form, vars):
    if vars['cif_or_pdb'] == "pdb":
        vars['pdb_object'] = pdbParser()
    else:
        vars['pdb_object'] = cifParser()
    vars['pdb_object'].read(vars['pdb_file_name'], form['PDB_chain'], form['DNA_AA'])

    #[vars['SEQRES_seq'], vars['ATOM_seq'], vars['ATOM_without_X_seq']] = get_seqres_atom_seq(vars['pdb_object'], form['PDB_chain'], vars['pdb_file_name'])
    vars['SEQRES_seq'] = vars['pdb_object'].get_SEQRES()
    All_atoms = vars['pdb_object'].get_ATOM_withoutX()
    if form['PDB_chain'] in All_atoms:
        vars['ATOM_without_X_seq'] = All_atoms[form['PDB_chain']]
    else:
        LOGGER.error("The chain is not in the PDB. Select the PDB chain using the flag --chain")

    analyse_seqres_atom(form, vars)
    try:
        FAS = open(vars['protein_seq'], 'w')
    except:
        LOGGER.error("cannot open the file " + vars['protein_seq'] + " for writing!")

    # we write the sequence to a fasta file for the homologues search
    # we save the name of the quey string for rate4site
    if vars['SEQRES_seq'] == "":
        vars['query_string'] = "Input_seq_ATOM_" + form['PDB_chain']
        vars['protein_seq_string'] = vars['ATOM_without_X_seq']
        FAS.write(">" + vars['query_string'] + "\n" + vars['ATOM_without_X_seq'])
    else:
        vars['query_string'] = "Input_seq_SEQRES_" + form['PDB_chain']
        vars['protein_seq_string'] = vars['SEQRES_seq']
        FAS.write(">" + vars['query_string'] + "\n" + vars['SEQRES_seq'])
    FAS.close()

class pdbParser:

    def __init__(self):

        self.SEQRES = ""
        self.ATOM = ""
        self.ATOM_withoutX = {}
        self.type = ""
        self.MODIFIED_COUNT = 0
        self.MODIFIED_LIST = ""
        self.positions = {}
        self.max_res_details = 0
        self.num_known_atoms = 0
        self.num_known_seqs = 0




    #def read(self, file, query_chain, DNA_AA, atom_position_filename):
    def read(self, file, query_chain, DNA_AA):

        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "5MU" : "t", "N" : "n"}
        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "N" : "n"}
        conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "A", "T" : "T", "C" : "C", "G" : "G", "U" : "U", "I" : "I", "DA" : "A", "DT" : "T", "DC" : "C", "DG" : "G", "DU" : "U", "DI" : "I", "5CM" : "C", "N" : "N"}
        modified_residues = {"MSE" : "MET", "MLY" : "LYS", "HYP" : "PRO", "CME" : "CYS", "CGU" : "GLU", "SEP" : "SER", "KCX" : "LYS", "MLE" : "LEU", "TPO" : "THR", "CSO" : "CYS", "PTR" : "TYR", "DLE" : "LEU", "LLP" : "LYS", "DVA" : "VAL", "TYS" : "TYR", "AIB" : "ALA", "OCS" : "CYS", "NLE" : "LEU", "MVA" : "VAL", "SEC" : "CYS", "PYL" : "LYS"}
        localMODRES = {}
        FIRST = [] # first residue in chain
        fas_pos = 0
        chain = "" # current chain
        ENDS = [] # end of chain reached and remaining HETATM should be skipped
        last_residue_number = ""

        if DNA_AA == "Nuc":

            UnknownChar = "N"

        else:

            UnknownChar = "X"
                        
        # open file to read MODRES
        try:

            PDBFILE = open(file, 'r')

        except:

            return 0

        try:

            MODRES_FILE = open(file + ".MODRES", 'w')

        except:

            return 0

        # read the MODRES
        line = PDBFILE.readline()
        while line != "" and not re.match(r'^ATOM', line):

            if re.match(r'^MODRES', line):

                MODRES = line[12:15].strip() # strip spaces to support NUC
                CHAIN = line[16:17]
                if CHAIN == " ":
                    
                    CHAIN = "NONE"
                    
                # we only look at the query chain
                if CHAIN != query_chain:
                    
                    line = PDBFILE.readline()
                    continue
                
                RES = line[24:27].strip() # strip spaces to support NUC

                if not MODRES in localMODRES:

                    localMODRES[MODRES] = RES
                    MODRES_FILE.write(MODRES + "\t" + RES + "\n")

                elif localMODRES[MODRES] != RES:

                    localMODRES[MODRES] = "" # two different values to the same residue

            line = PDBFILE.readline()


        MODRES_FILE.close()
        PDBFILE.close()
        
        # reopen file to read all the file
        try:

            PDBFILE = open(file, 'r')

        except:

            return 0

        line = PDBFILE.readline()
        while line != "":

            line = line.strip()

            if re.search(r'^SEQRES', line): # SEQRES record

                chain_seqres = line[11:12] # get chain id
                if chain_seqres == " ":

                    chain_seqres = "NONE"

                # we skip the chain if it is not the query
                if query_chain != chain_seqres:
                    
                    line = PDBFILE.readline()
                    continue

                # convert to one letter format
                for acid in line[19:70].split():

                    # regular conversion
                    if acid in conversion_table:

                        # add to chain
                        self.SEQRES += conversion_table[acid]
                        self.num_known_seqs += 1

                    # modified residue
                    else:

                        # count this modified residue
                        self.MODIFIED_COUNT += 1

                        # check if residue is identified
                        if acid in modified_residues and modified_residues[acid] in conversion_table:

                            self.SEQRES += conversion_table[modified_residues[acid]]
                            self.num_known_seqs += 1

                            # add to modified residue list
                            if not acid + " > " in self.MODIFIED_LIST:

                                self.MODIFIED_LIST += acid + " > " + conversion_table[modified_residues[acid]] + "\n"

                        elif acid in localMODRES and localMODRES[acid] != "" and localMODRES[acid] in conversion_table:

                            self.SEQRES += conversion_table[localMODRES[acid]]
                            self.num_known_seqs += 1

                            # add to modified residue list
                            if not acid + " > " in self.MODIFIED_LIST:

                                self.MODIFIED_LIST += acid + " > " + conversion_table[localMODRES[acid]] + "\n"

                        else:

                            # set residue name to X or N
                            self.SEQRES += UnknownChar

                            # add message to front of modified residue list
                            modified_changed_to_X_or_N_msg = "Modified residue(s) in this chain were converted to the one letter representation '" + UnknownChar + "'\n"

                            if not "Modified residue" in self.MODIFIED_LIST:

                                self.MODIFIED_LIST = modified_changed_to_X_or_N_msg + self.MODIFIED_LIST

            elif re.search(r'^ATOM', line):

                # extract atom data
                res = line[17:20].strip() # for DNA files there is only one or two letter code
                chain = line[21:22]
                pos = line[22:27].strip()

                if chain == " ":

                    chain = "NONE"

                # find modified residue
                if res in modified_residues:
                    
                    mod_res = modified_residues[res]
                    
                elif res in localMODRES and localMODRES[res] != "" and localMODRES[res] in conversion_table:
                    
                    mod_res = localMODRES[res]

                else:
                    
                    mod_res = res
                  
                # convert residue to one letter
                if mod_res in conversion_table:
                    
                    oneLetter = conversion_table[mod_res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    last_pos = pos
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = PDBFILE.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = PDBFILE.readline()
                    continue 
                  
                self.num_known_atoms += 1
                
                # writing atom position file
                fas_pos += 1
                res_details = "%s:%s:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                #CORR.write("%s\t%d\t%s\n" %(res, fas_pos, pos))
                        
                #residue_number = int(line[22:26].strip())
                        
                # check type 
                if self.type == "":

                    if len(mod_res) < 3:

                        self.type = "Nuc"

                    else:

                        self.type = "AA"
                """    
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_residue_number < residue_number:
                    
                    while residue_number != last_residue_number + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_residue_number += 1
                              
                self.ATOM += oneLetter
                last_residue_number = residue_number
                """
            elif re.search(r'^HETATM', line):

                # extract hetatm data
                res = line[17:20].strip() # for DNA files there is only one or two letter code
                chain = line[21:22]
                pos = line[22:27].strip()

                if chain == " ":

                    chain = "NONE"
                    
                if chain in ENDS:
                         
                    line = PDBFILE.readline()
                    continue 
                    
                # find modified residue
                if res in modified_residues:
                    
                    mod_res = modified_residues[res]
                    
                elif res in localMODRES and localMODRES[res] != "" and localMODRES[res] in conversion_table:
                    
                    mod_res = localMODRES[res]

                else:
                    
                    mod_res = res
                  
                # convert residue to one letter
                if mod_res in conversion_table:
                    
                    oneLetter = conversion_table[mod_res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    last_pos = pos
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = PDBFILE.readline()
                    continue                      
                
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = PDBFILE.readline()
                    continue 
                
                self.num_known_atoms += 1
                
                # writing atom position file
                fas_pos += 1
                res_details = "%s:%s:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                """                        
                residue_number = int(line[22:26].strip())
                
                if FIRST[chain]:
                    
                    last_residue_number = residue_number
                    FIRST[chain] = False

                elif last_residue_number < residue_number:
                    
                    while residue_number != last_residue_number + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_residue_number += 1
                              
                self.ATOM += oneLetter
                last_residue_number = residue_number   
                """
            elif re.search(r'^TER', line):
                
                if not chain in ENDS:
                    
                    ENDS.append(chain)
                
            line = PDBFILE.readline()

        PDBFILE.close()
        #CORR.close()
        return 1

    def get_num_known_atoms(self):
        
        return self.num_known_atoms

    def get_num_known_seqs(self):
        
        return self.num_known_seqs

    def get_max_res_details(self):
        
        return self.max_res_details

    def get_positions(self):
        
        return self.positions

    def get_type(self):

        return self.type

    def get_SEQRES(self):

        return self.SEQRES

    def get_ATOM_withoutX(self):

        return self.ATOM_withoutX

    def get_MODIFIED_COUNT(self):
        

        return self.MODIFIED_COUNT
        


    def get_MODIFIED_LIST(self):

        return self.MODIFIED_LIST




class cifParser:

    def __init__(self):

        self.SEQRES = ""
        #self.ATOM = ""
        self.ATOM_withoutX = {}
        self.type = ""
        self.MODIFIED_COUNT = 0
        self.MODIFIED_LIST = ""
        self.positions = {}
        self.max_res_details = 0
        self.auth_seq_id_column = 0
        self.auth_comp_id_column = 0
        self.auth_asym_id_column = 0
        self.B_iso_or_equiv = 0


    #def read(self, file, query_chain, DNA_AA, atom_position_filename):
    def read(self, file, query_chain, DNA_AA):

        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "N" : "n"}
        conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "A", "T" : "T", "C" : "C", "G" : "G", "U" : "U", "I" : "I", "DA" : "A", "DT" : "T", "DC" : "C", "DG" : "G", "DU" : "U", "DI" : "I", "5CM" : "C", "N" : "N"}
        #modified_residues = {"MSE" : "MET", "MLY" : "LYS", "HYP" : "PRO", "CME" : "CYS", "CGU" : "GLU", "SEP" : "SER", "KCX" : "LYS", "MLE" : "LEU", "TPO" : "THR", "CSO" : "CYS", "PTR" : "TYR", "DLE" : "LEU", "LLP" : "LYS", "DVA" : "VAL", "TYS" : "TYR", "AIB" : "ALA", "OCS" : "CYS", "NLE" : "LEU", "MVA" : "VAL", "SEC" : "CYS", "PYL" : "LYS"}
        
        # find the portion of the file that contains the seqres
        SEQRES_string = ""
        SEQRES_string_found = False
        in_fasta = False
        fas_pos = 0
        last_pos = 0
        current_chain = ""
        hetatm = "" # the part of the sequence that in the HETATM lines
        hetatm_withoutX = "" # X is not added to fill the breaks in the sequence
        hetatm_pos = {} # the positions of the residues in the HETATM
        hetatm_max_res_details = 0 # maximum length of the details of the residues in the HETATM
        
        if DNA_AA == "Nuc":

            UnknownChar = "N"

        else:

            UnknownChar = "X"
            
        try:

            CIF = open(file, 'r')

        except:

            return 0
        
        line = CIF.readline()
        while line != "":

            if re.match(r'^_entity_poly.entity_id', line):

                while line != "":

                    if ';' in line:

                        in_fasta = not in_fasta

                    line = line.replace(";", "")

                    if '#' in line:

                        # end of _entity_poly.entity_id
                        SEQRES_string_found = True
                        break

                    elif in_fasta:

                        # delete white spaces in fasta
                        SEQRES_string += line.strip()

                    else:

                        SEQRES_string += line

                    line = CIF.readline()

            if SEQRES_string_found:

                break

            line = CIF.readline()

        if re.match(r'^_entity_poly.entity_id\s+1', SEQRES_string):

            # one seqres
            match1 = re.search(r'_entity_poly.pdbx_seq_one_letter_code_can\s+(\S+)', SEQRES_string)
            seqres = match1.group(1)

            match2 = re.search(r'_entity_poly.pdbx_strand_id\s+(\S+)', SEQRES_string)
            for chain in (match2.group(1)).split(','):

                if chain == query_chain:
                    
                    self.SEQRES = seqres

        else:

            # more than one seqres
            SEQRES_substrings = re.split(r'\d+\s+\'?poly', SEQRES_string)

            POLY = open("poly", 'w')
            for string in SEQRES_substrings:

                POLY.write(string + "\n>\n")

            POLY.close()

            SEQRES_substrings = SEQRES_substrings[1:] # delete titles
            for substring in SEQRES_substrings:

                words = substring.split()
                for chain in (words[5]).split(','):
                    
                    if chain == query_chain:

                        self.SEQRES = words[4]
                    
        number_of_columns = 0
        # we find which columns has what value
        auth_seq_id_column = 0
        auth_comp_id_column = 0
        auth_asym_id_column = 0
        B_iso_or_equiv = 0
        found_auth_seq_id_column = False
        found_auth_comp_id_column = False
        found_auth_asym_id_column = False
        found_B_iso_or_equiv = False
        while line != "":

            line = line.strip()

            if line == "_atom_site.B_iso_or_equiv":

                found_B_iso_or_equiv = True

            if line == "_atom_site.auth_seq_id":

                found_auth_seq_id_column = True

            if line == "_atom_site.auth_comp_id":

                found_auth_comp_id_column = True

            if line == "_atom_site.auth_asym_id":

                found_auth_asym_id_column = True

            if not re.match(r'^_atom_site.', line) and found_B_iso_or_equiv and found_auth_seq_id_column and found_auth_comp_id_column and found_auth_asym_id_column:

                # we identified the necessary columns
                break

            if found_B_iso_or_equiv:

                B_iso_or_equiv -= 1

            if found_auth_seq_id_column:

                auth_seq_id_column -= 1

            if found_auth_comp_id_column:

                auth_comp_id_column -= 1

            if found_auth_asym_id_column:

                auth_asym_id_column -= 1

            line = CIF.readline()

        FIRST = []
        while line.strip() != "":

            words = line.split()
            if words[0] == "ATOM" and words[1].isnumeric():

                number_of_columns = len(words)

                # extract atom data
                pos = int(words[auth_seq_id_column])
                res = words[auth_comp_id_column]
                chain = words[auth_asym_id_column]

                # if HETATM is not in the end of the chain we add it to the sequence
                if chain == current_chain:
                    
                    self.ATOM_withoutX[chain] += hetatm_withoutX

                else:
                    
                    current_chain = chain
                    
                hetatm_withoutX = ""
                    
                # convert residue to one letter
                if res in conversion_table:
                    
                    oneLetter = conversion_table[res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = CIF.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    #hetatm = ""
                    line = CIF.readline()
                    continue 
                """ 
                else:
                    
                    self.ATOM += hetatm
                    hetatm = ""
                """  
                    

                # writing atom position file
                
                self.positions.update(hetatm_pos)
                if hetatm_max_res_details > self.max_res_details:
                    
                    self.max_res_details = hetatm_max_res_details
                    
                hetatm_pos = {}
                hetatm_max_res_details = 0
                
                #CORR.write(hetatm_pos)
                #hetatm_pos = ""

                
                fas_pos += 1
                
                res_details = "%s:%d:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                #CORR.write("%s\t%d\t%s\n" %(res, fas_pos, pos))
                                                
                # check type 
                if self.type == "":

                    if len(res) < 3:

                        self.type = "Nuc"

                    elif len(res) == 3:

                        self.type = "AA"
                """           
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_pos < pos:
                    
                    while pos != last_pos + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_pos += 1
                            
                self.ATOM += oneLetter
                """
                last_pos = pos
                
            elif words[0] == "HETATM" and words[1].isnumeric():
                

                # extract atom data
                pos = int(words[auth_seq_id_column])
                res = words[auth_comp_id_column]
                chain = words[auth_asym_id_column]

                    
                # convert residue to one letter
                if res in conversion_table:
                    
                    oneLetter = conversion_table[res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = ""
                    hetatm_withoutX = oneLetter
                    
                elif pos != last_pos:
                    
                    hetatm_withoutX += oneLetter

                else:

                    line = CIF.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = CIF.readline()
                    continue 
                
                # writing atom position file
                fas_pos += 1
                
                res_details = "%s:%d:%s" %(res, pos, chain)
                hetatm_pos[fas_pos] = res_details
                if len(res_details) > hetatm_max_res_details:
                    
                    hetatm_max_res_details = len(res_details)
                    
                #hetatm_pos += "%s\t%d\t%s\n" %(res, fas_pos, pos)
                                                
                # check type 
                if self.type == "":

                    if len(res) < 3:

                        self.type = "Nuc"

                    elif len(res) == 3:

                        self.type = "AA"
                """            
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_pos < pos:
                    
                    while pos != last_pos + 1: # For Disorder regions
                    
                        hetatm += UnknownChar
                        last_pos += 1
                            
                hetatm += oneLetter
                """
                last_pos = pos    
                
            line = CIF.readline()

        CIF.close()
        #CORR.close()
        
        self.auth_seq_id_column = number_of_columns + auth_seq_id_column
        self.auth_comp_id_column = number_of_columns + auth_comp_id_column
        self.auth_asym_id_column = number_of_columns + auth_asym_id_column
        self.B_iso_or_equiv = number_of_columns + B_iso_or_equiv
        
        
        return 1
        

    def get_max_res_details(self):
        
        return self.max_res_details

    def get_columns(self):
        
        return self.auth_seq_id_column, self.auth_comp_id_column, self.auth_asym_id_column, self.B_iso_or_equiv

    def get_type(self):

        return self.type

    def get_SEQRES(self):

        return self.SEQRES

    def get_ATOM_withoutX(self):

        return self.ATOM_withoutX

    def get_MODIFIED_COUNT(self):
        

        return self.MODIFIED_COUNT
        


    def get_MODIFIED_LIST(self):

        return self.MODIFIED_LIST


    def get_positions(self):
        
        return self.positions

def analyse_seqres_atom(form, vars):
    # there is no ATOM field in the PDB
    if vars['ATOM_without_X_seq'] == "":
        LOGGER.error("There is no ATOM derived information in the PDB file.<br>Please refer to the OVERVIEW for detailed information about the PDB format.")
    # there is no SEQRES field in the PDB
    if vars['SEQRES_seq'] == "":
        msg = "Warning: There is no SEQRES derived information in the PDB file. The calculation will be based on the ATOM derived sequence. "
        if vars['running_mode'] == "_mode_pdb_no_msa":
            msg += "If this sequence is incomplete, we recommend to re-run the server using an external multiple sequence alignment file, which is based on the complete protein sequence."
        LOGGER.info("analyse_seqres_atom : There is no SEQRES derived information in the PDB file.\n")

    if form['DNA_AA'] == "AA":
        # check if seqres contains nucleic acid
        if vars['pdb_object'].get_type() == "Nuc":
            LOGGER.error("The selected chain: " + form['PDB_chain'] + " contains nucleic acid, and you have selected amino acid")
    else:
        # check if seqres contains amino acid
        #type_SEQRES = vars['pdb_object'].get_type_SEQRES()
        #if form['PDB_chain'] in type_SEQRES and type_SEQRES[form['PDB_chain']] == "AA":
        if vars['pdb_object'].get_type() == "AA":
            LOGGER.error("The selected chain: " + form['PDB_chain'] + " contains amino acid, and you have selected nucleic acid")

    # if modified residues exists, print them to the screen
    MODIFIED_COUNT = vars['pdb_object'].get_MODIFIED_COUNT()
    if MODIFIED_COUNT > 0:
        if form['DNA_AA'] == "AA":
            if len(vars['SEQRES_seq']) > 0 and MODIFIED_COUNT / len(vars['SEQRES_seq']) > GENERAL_CONSTANTS.MAXIMUM_MODIFIED_PERCENT:
                LOGGER.info("MODIFIED_COUNT %d\nSEQRES_seq %s\n" %(MODIFIED_COUNT, vars['SEQRES_seq']))
                LOGGER.error("Too many modified residues were found in SEQRES field; %0.3f%% of the residues are modified, the maximum is %0.3f%%." %(MODIFIED_COUNT / len(vars['SEQRES_seq']) ,GENERAL_CONSTANTS.MAXIMUM_MODIFIED_PERCENT))
            LOGGER.info("analyse_seqres_atom : modified residues found\n")
            LOGGER.info("Please note: Before the analysis took place, modified residues read from SEQRES field were converted back to the original residues:\n" + vars['pdb_object'].get_MODIFIED_LIST() + ".")
        else:
            LOGGER.info("analyse_seqres_atom : modified residues found\n")
            LOGGER.info("Please note: Before the analysis took place, modified nucleotides read from SEQRES field were converted back to the original nucleotides:\n" + vars['pdb_object'].get_MODIFIED_LIST() + ".")

def no_MSA(form, vars):
    # if there is pdb : we compare the atom and seqres
    if vars['running_mode'] ==  "_mode_pdb_no_msa" and ('SEQRES_seq' in vars and len(vars['SEQRES_seq']) > 0):
        # align seqres and pdb sequences
        compare_atom_seqres_or_msa("SEQRES", form, vars)

    vars['max_homologues_to_display'] = GENERAL_CONSTANTS.BLAST_MAX_HOMOLOGUES_TO_DISPLAY

    blast_hash = {}
    #vars['hit_min_length'] = GENERAL_CONSTANTS.FRAGMENT_MINIMUM_LENGTH
    run_search(vars['BLAST_out_file'], form, vars)

    # choosing homologs, create fasta file for all legal homologs
    cd_hit_hash = {}
    #vars['hit_redundancy'] = float(form['MAX_REDUNDANCY']) # Now taken as argument from user #OLD: #CONSURF_CONSTANTS.FRAGMENT_REDUNDANCY_RATE
    vars['hit_overlap'] = GENERAL_CONSTANTS.FRAGMENT_OVERLAP
    #vars['min_num_of_hits'] = GENERAL_CONSTANTS.MINIMUM_FRAGMENTS_FOR_MSA
    vars['low_num_of_hits'] = GENERAL_CONSTANTS.LOW_NUM_FRAGMENTS_FOR_MSA
    vars['HITS_fasta_file'] = "query_homolougs.txt"
    vars['HITS_rejected_file'] = "query_rejected_homolougs.txt"

    choose_homologoues_from_search_with_lower_identity_cutoff(form['Homolog_search_algorithm'], len(vars['protein_seq_string']), vars['hit_redundancy'], vars['hit_overlap'], vars['hit_min_length'], float(form['MIN_IDENTITY']), vars['min_num_of_hits'], vars['BLAST_out_file'], vars['HITS_fasta_file'], vars['HITS_rejected_file'], blast_hash, form['DNA_AA'], form, vars)
    vars['cd_hit_out_file'] = "query_cdhit.out"
    vars['unique_seqs'] = cluster_homologoues(cd_hit_hash, form, vars)
    LOGGER.info("num_of_unique_seq: %d\n" % vars['unique_seqs'])

    add_sequences_removed_by_cd_hit_to_rejected_report(vars['cd_hit_out_file'] + ".clstr", vars['HITS_rejected_file'], vars['num_rejected_homologs'], vars)
    choose_final_homologoues(blast_hash, cd_hit_hash, float(form['MAX_NUM_HOMOL']) -1, form['best_uniform_sequences'], vars['FINAL_sequences'], vars['HITS_rejected_file'], vars['num_rejected_homologs'], vars)
    if form['DNA_AA'] == "Nuc":
        # convert rna to dna
        LOGGER.info("convert_rna_to_dna(%s, %s)\n" %(vars['FINAL_sequences'], vars['FINAL_sequences'] + ".dna"))
        ans = convert_rna_to_dna(vars['FINAL_sequences'], vars['FINAL_sequences'] + ".dna")
        if ans[0] == "OK":
            vars['FINAL_sequences'] += ".dna"
            LOGGER.info("Seqs with u or U: " + str(ans[1]))
            for seq in ans[1]:
                LOGGER.info("Warnning: The seqeunce '" + seq + "' contains a 'U' replaced by 'T'")
        else:
            LOGGER.error(f"convert_rna_to_dna failed: {ans}")

    LOGGER.info("make_sequences_file_HTML(%s, %s)\n" %(vars['FINAL_sequences'], vars['FINAL_sequences_html']))
    make_sequences_file_HTML(vars['FINAL_sequences'], vars['FINAL_sequences_html'])

    # we save to copies of the msa, one in fasta format and another in clustal format.
    #vars['msa_fasta'] = "msa_fasta.aln"
    #vars['msa_clustal'] = "msa_clustal.aln"
    create_MSA(form, vars)
    vars['msa_SEQNAME'] = vars['query_string']

def compare_atom_seqres_or_msa(what_to_compare, form, vars):

    # in case there is a msa and pdb, we check the similarity between the atom and the msa sequences
    # in case there is no msa and there are both atom and seqres sequences, we check the similarity between them

    pairwise_aln = "PDB_" + what_to_compare + ".aln"
    atom_length = len(vars['ATOM_without_X_seq'])
    alignment_score = 0
    other_query_length = len(vars['protein_seq_string'])
    query_line = {}
    atom_line = "sequence extracted from the ATOM field of the PDB file"
    query_line['SEQRES'] = "sequence extracted from the SEQRES field of the PDB file"
    query_line['MSA'] = "sequence extracted from the MSA file"

    # compare the length of sequences. output a message accordingly
    if other_query_length != 0 and other_query_length < atom_length:
        LOGGER.info("The %s is shorter than the %s. The %s sequence has %d residues and the ATOM sequence has %d residues. The calculation continues nevertheless." %(query_line[what_to_compare],atom_line ,what_to_compare, other_query_length, atom_length))

    if atom_length < other_query_length:
        if atom_length < other_query_length * 0.2:
            LOGGER.info("Warning: The %s is significantly shorter than the %s. The %s sequence has %d residues and the ATOM sequence has only %d residues. The calculation continues nevertheless." %(atom_line, query_line[what_to_compare], what_to_compare, other_query_length, atom_length))
        else:
            LOGGER.info("The %s is shorter than the %s. The %s sequence has %d residues and the ATOM sequence has %d residues. The calculation continues nevertheless." %(atom_line, query_line[what_to_compare], what_to_compare, other_query_length, atom_length))

    # match the sequences 
    LOGGER.info(f"compare_atom_seqres_or_msa : Align ATOM and {what_to_compare} sequences\n")
    [
        vars['seqres_or_msa_seq_with_gaps'], 
        vars['ATOM_seq_with_gaps'], 
        alignment_score
        ] = pairwise_alignment(vars['protein_seq_string'], vars['ATOM_without_X_seq'], pairwise_aln, what_to_compare, form, vars)

    if alignment_score < 100:
        if alignment_score < 30:
            LOGGER.error(f"The Score of the alignment between the {query_line[what_to_compare]} and the {atom_line} is ONLY {alignment_score}% identity. See pairwise alignment.")
        else:
            LOGGER.info("The Score of the alignment between the %s and the %s is %d%% identity. The calculation continues nevertheless." %(query_line[what_to_compare], atom_line, alignment_score))

def run_search(Search_Out_File, form, vars):

    # search for homologs
    cmd = ""
    if form['DNA_AA'] == "AA":
        if form['Homolog_search_algorithm'] == "BLAST":
            cmd = "psiblast -outfmt 5 -query %s -evalue %s -db %s -num_iterations %s -out %s\n" %(vars['protein_seq'], form['E_VALUE'], vars['protein_db'], form['ITERATIONS'], Search_Out_File)
        elif form['Homolog_search_algorithm'] == "HMMER":
            cmd = "jackhmmer --notextw -N %s --domE %s -E %s --incE %s --cpu 4 %s  %s > %s\n" %(form['ITERATIONS'], form['E_VALUE'], form['E_VALUE'], form['E_VALUE'],  vars['protein_seq'], vars['protein_db'], Search_Out_File)
        elif form['Homolog_search_algorithm'] == "MMSEQS2":
            cmd = "mmseqs createdb  %s query.fa\nmmseqs search query.fa %s alnNew temp -a 1 --max-seqs 10000000 -s 5.7 --min-seq-id %.2f --max-seq-id %.2f -c %s --filter-hits 1\nmmseqs convertalis query.fa %s alnNew %s --format-output \"target,evalue,taln,tstart,tend,pident,theader\"\n" %(vars['protein_seq'], vars['protein_db'], float(form['MIN_IDENTITY']) / 100, float(form['MAX_REDUNDANCY']) / 100, vars['hit_min_length'], vars['protein_db'], Search_Out_File)
        else: # if form['Homolog_search_algorithm'] == "CS_BLAST"
            cmd = "%s -i %s -e %s -d %s -D %s/K4000.lib -j %s -v %s -b %s -o %s -m 7 -F F --blast-path %s -T F\n" %(GENERAL_CONSTANTS.CS_BLAST, vars['protein_seq'], form['E_VALUE'], vars['protein_db'], GENERAL_CONSTANTS.CS_BLAST_DATA, form['ITERATIONS'], vars['max_homologues_to_display'], vars['max_homologues_to_display'], Search_Out_File, GENERAL_CONSTANTS.BLAST_PATH)
    else:
        if form['Homolog_search_algorithm'] == "HMMER":
            cmd = "nhmmer --notextw --domE %s -E %s --cpu 4 %s %s > %s\n" %(form['E_VALUE'], form['E_VALUE'], vars['protein_seq'], vars['protein_db'], Search_Out_File)
        else:
            #cmd = "%s -p blastn -m 7 -i %s -e %s -d %s -v %s -b %s -o %s -F F -T F\n" %(GENERAL_CONSTANTS.BLASTALL, vars['protein_seq'], form['E_VALUE'], GENERAL_CONSTANTS.NR_NUC_DB, vars['max_homologues_to_display'], vars['max_homologues_to_display'], Search_Out_File)
            cmd = "blastn -outfmt 5 -query %s -evalue %s -db %s -num_alignments %s -out %s -dust no -task blastn\n" %(vars['protein_seq'], form['E_VALUE'], vars['protein_db'], vars['max_homologues_to_display'], Search_Out_File)

    LOGGER.info("run_search : running: " + cmd + "\n")
    submit_job_to_Q(form['Homolog_search_algorithm'], cmd)

    if not os.path.exists(Search_Out_File) or os.path.getsize(Search_Out_File) == 0:
        LOGGER.error(f"run_search : run of search fail. {Search_Out_File} is zero or not exists")

def submit_job_to_Q(job_name_prefix, cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()


def choose_homologoues_from_search_with_lower_identity_cutoff(searchType, query_seq_length, redundancyRate, frag_overlap, min_length_percent, min_id_percent, min_num_of_homologues, search_output, fasta_output, rejected_seqs, ref_search_hash, Nuc_or_AA, 
    form, vars):

    # searchType: HMMER, BLAST or MMseqs2
    # query_seq_length: Length of the query sequence
    # redundancyRate: The allowed similarity between the query and the hit
    # frag_overlap: The allowed overlap between the hits
    # min_length_percent: The hit can't be smaller than this percent of the query
    # min_id_percent: The minimum similarity between the query and the hit
    # min_num_of_homologues: Minimum number of homologs
    # search_output: Raw homolog search output
    # fasta_output: Accepted hits
    # rejected_seqs: Rejected hits
    # ref_search_hash: Hash with the evalues of the accepted hits. This is later used when choosing the final hits after cid-hit
    # Nuc_or_AA: Amino or nucleic acid
    # animal_name_pattern: This is used to find the animal name in the description of the hit
    
    LOGGER.info("choose_homologoues_from_search_with_lower_identity_cutoff(%s, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);\n" %(searchType, query_seq_length, redundancyRate, frag_overlap, min_length_percent, min_id_percent, min_num_of_homologues, search_output, fasta_output, rejected_seqs, ref_search_hash, Nuc_or_AA))

    # Defining the minimum length a homologue should have
    # 60% the query's length
    min_length = query_seq_length * min_length_percent

    # Reading blast/hmmer output and collect the homologues
    # Printing the selected homologues to a file and insert the e-value info to hash
    try:
        OUT_REJECT = open(rejected_seqs, 'w')
    except:
        LOGGER.error("choose_homologoues_from_search_with_lower_identity_cutoff : can't open file " + rejected_seqs + " for writing\n")
    try:
        OUT = open(fasta_output, 'w')
    except:
        LOGGER.error("choose_homologoues_from_search_with_lower_identity_cutoff : can't open file " + fasta_output + " for writing\n")
    try:
        RAW_OUT = open(search_output, 'r')
    except:
        LOGGER.error("choose_homologoues_from_search_with_lower_identity_cutoff : can't open file " + search_output + " for reading\n")

    num_homologoues = 0
    num_rejected = 1
    final_num_homologues = 0
    OUT_REJECT.write("\tSequences rejected for being too short, too similar, or not similar enough to the query sequence.\n\n")
    if searchType == "HMMER":
        # Moving the start position to the last round of the output
        position = 0
        line = RAW_OUT.readline()
        while line != "":
            if line.rstrip() == "[ok]":
                RAW_OUT.seek(position)
                break
            else:
                match =re.match(r'^@@ Round:', line)
                if match:
                    position = RAW_OUT.tell()
            line = RAW_OUT.readline()
        # we skip the beginning
        line = RAW_OUT.readline()
        while line != "":
            if re.match(r'^>>\s*(\S+)\s+(.*)', line):
                break
            else:
                 line = RAW_OUT.readline()
        # we process the output
        while line != "":
            # each seq may have more than one domain
            match1 = re.match(r'^>>\s*(\S+)\s+(.*)', line)
            if match1:
                # new seq found
                num_homologoues += 1
                seq_name = match1.group(1)
                regex = r"^\s*" + re.escape(seq_name) + r"\s*(\S+)\s+(\S+)\s+(\S+)"
                seq_name = change_name(seq_name)
                seq_description = match1.group(2).strip()
                #animal_name = get_animal_name(animal_name_pattern, seq_description)
                
                domains_ends = [] # The array will hold the beginning and ending of the domains
                line = RAW_OUT.readline()
                # we extract data from the domains
                while line != "":
                    if Nuc_or_AA == "AA":
                        match2 = re.match(r'.*E-value:\s+(\S+)\s*', line)
                    else:
                        match2 = re.match(r'^\s+!\s+\S+\s+\S+\s+(\S+)', line)
                    if match2:
                        # new domain found
                        seq_eval = match2.group(1)
                        prev_line = line # the line between the db seq and the query seq contains the residues that appear in both
                        line = RAW_OUT.readline()
                        while line != "":
                            match3 = re.match(regex, line)
                            if match3:
                                seq_beg = int(match3.group(1))
                                seq = match3.group(2) # the seq in fasta with gaps
                                seq_end = int(match3.group(3))
                                seq_ident = (float(count_letters(prev_line)) / float(len(seq))) * 100
                                seq = re.sub(r'-', "", seq) # delete gaps
                                seq_frag_name = "%s|%d_%d|%s|%s" %(seq_name, seq_beg, seq_end, seq_eval, seq_description)

                                # deciding if we take the fragment
                                # in case there is already a fragemnt with the same name, we do another validity test for overlapping
                                if len(domains_ends) > 0:
                                    # there is more than one domain. check if there is overlap
                                    ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, seq_ident, seq, seq_name, Nuc_or_AA)
                                    if ans == "yes":
                                        ans = check_if_no_overlap(frag_overlap, domains_ends, seq_beg, seq_end)
                                else:
                                    ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, seq_ident, seq, seq_name, Nuc_or_AA)

                                # if the sequence is valid, insert it to the hash
                                if ans == "yes":
                                    # we save the ends to check for overlap with other domains
                                    domains_ends.append([seq_beg, seq_end])
                                    final_num_homologues += 1
                                    OUT.write(">%s\n%s\n" %(seq_frag_name, seq))
                                    ref_search_hash[seq_frag_name] = float(seq_eval)
                                else:
                                    OUT_REJECT.write("%d Fragment %s rejected: %s\n" %(num_rejected, seq_frag_name, ans))
                                    num_rejected += 1
                                # end of domain, move to next domain
                                break
                            prev_line = line
                            line = RAW_OUT.readline()
                    elif line[:2] == ">>":
                        # end of seq, move to next seq
                        break
                    else:
                        line = RAW_OUT.readline()
    elif searchType == "MMSEQS2":
        line = RAW_OUT.readline()
        while line != "":
            words = line.split()
            if len(words) > 0:
                # new seq found
                num_homologoues += 1
                seq_name = change_name(words[0])
                description = ""
                for word in words[5:]: # get description
                    description += " " + word
                #animal_name = get_animal_name(animal_name_pattern, description)
                #sequences = [] # The array will hold the beginning and ending of the domains
                seq_eval = words[1]
                seq_beg = int(words[3])
                seq = words[2] # the seq in fasta with gaps
                seq_end = int(words[4])
                seq_ident = float(words[5])
                seq = re.sub(r'-', "", seq) # delete gaps
                seq_frag_name = "%s|%d_%d|%s|%s" %(seq_name, seq_beg, seq_end, seq_eval, description)
                #seq_name_exists = "no"

                # deciding if we take the fragment
                ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, seq_ident, seq, seq_name, Nuc_or_AA)
                # after taking the info, check if the currecnt sequence is valid. If so - insert it to the hash
                if ans == "yes":
                    final_num_homologues += 1
                    OUT.write(">%s\n%s\n" %(seq_frag_name, seq))
                    ref_search_hash[seq_frag_name] = seq_eval
                else:
                    OUT_REJECT.write("%d Fragment %s rejected: %s\n" %(num_rejected, seq_frag_name, ans))
                    num_rejected += 1
            line = RAW_OUT.readline()
    else:
        # BLAST
        searchio = SearchIO.parse(search_output, "blast-xml")
        last_blast_round = ""
        for qresult in searchio:
            last_blast_round = qresult
        for hit in last_blast_round:
            seq_description = hit.description
            #animal_name = get_animal_name(animal_name_pattern, hit.description)
            domains_ends = [] # The array will hold the beginning and ending of the domains
            s_name = change_name(hit.id)
            #s_name = hit.id
            #seq_name_exists = "no"
            for hsp in hit:
                # hsp is the next available High Scoring Pair, Bio::Search::HSP::HSPI object or null if finished
                num_homologoues += 1
                # extracting relevant details from the fragment
                [s_beg, s_end] = hsp.hit_range
                s_beg += 1
                AAseq = str(hsp.hit.seq)
                AAseq = re.sub(r'-', "", AAseq)
                s_ident = (float(hsp.ident_num) / float(hsp.aln_span)) * 100
                s_eval = str(hsp.evalue)
                s_eval = re.sub(r',', "", s_eval)
                match = re.match(r'^e', s_eval)
                if match:
                    s_eval = "1" + s_eval
                seq_frag_name = "%s|%d_%d|%s|%s" %(s_name, s_beg, s_end, s_eval, seq_description)
                # deciding if we take the fragment
                if len(domains_ends) > 0:
                    #seq_name_exists = "yes"
                    #seq_details = sequences[s_name]
                    ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, s_ident, AAseq, s_name, Nuc_or_AA)
                    if ans == "yes":
                        ans = check_if_no_overlap(frag_overlap, domains_ends, s_beg, s_end)
                else:
                    ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, s_ident, AAseq, s_name, Nuc_or_AA)
                # if the sequence is valid, insert it to the hash
                if ans == "yes":
                    # we save the ends to check for overlap with other domains
                    domains_ends.append([s_beg, s_end])
                    final_num_homologues += 1
                    OUT.write(">%s\n%s\n" %(seq_frag_name, AAseq))
                    ref_search_hash[seq_frag_name] = float(s_eval)
                else:
                    OUT_REJECT.write("%d Fragment %s rejected: %s\n" %(num_rejected, seq_frag_name, ans))
                    num_rejected += 1
    OUT_REJECT.close()
    OUT.close()
    # Checking that the number of homologues found is legal

    message = ""
    if final_num_homologues == 1:
        message = "only one unique sequence "
    elif final_num_homologues < min_num_of_homologues:
        message = "only %d unique sequences " %final_num_homologues
    else:
        message ="ok"

    if message != "ok":
        if searchType != "HMMER":
            message += "were chosen from BLAST output. " 
        else:
            message += "were chosen from HMMER output. "
        message += "The minimal number of sequences required for the calculation is " + str(min_num_of_homologues) + "."
        message = "According to the parameters of this run, " + message + " You can try to: "  
        message += "Re-run the program with a multiple sequence alignment file of your own. Increase the Evalue. Decrease the Minimal %ID For Homologs"
        if int(form['ITERATIONS']) < 5:
            message += " Increase the number of " + form['Homolog_search_algorithm'] + " iterations."
        message += "\n"
        LOGGER.error("choose_homologoues_from_search_with_lower_identity_cutoff : " + message)

    vars['number_of_homologoues'] = num_homologoues
    vars['number_of_homologoues_before_cd-hit'] = final_num_homologues
    vars['num_rejected_homologs'] = num_rejected

def change_name(s_name):
    if s_name[:2] == "sp" or s_name[:2] == "tr":
        words = s_name.split("|")
        return  "up|" + words[1]
    elif s_name[:2] == "gi":
        words = s_name.split("|")
        return  "gi|" + words[1]
    elif s_name[:8] == "UniRef90":
        words = s_name.split("_")
        return "ur|" + words[1]
    elif "|" in s_name:
        words = s_name.split("|")
        return "up|" + words[1]
    else:
        return "gi|" + s_name    

def check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id, ident_percent, aaSeq, seqName, Nuc_or_AA):

    seq_length = len(aaSeq)

    if ident_percent >= redundancyRate:
        # the sequence identity is not too high
        return "identity percent %.2f is too big" %ident_percent

    if ident_percent < min_id:
        # the sequence identity is higher than the minium idnentity percent that was defined for homologus
        return "identity percent %.2f is too low (below %d)" %(ident_percent, min_id)

    elif seq_length < min_length:
        # the sequnece length is greater than the minimum sequence length
        return "the sequence length %d is too short. The minimum is %d" %(seq_length, min_length)

    return check_illegal_character(aaSeq, seqName, Nuc_or_AA)

def check_illegal_character(aaSeq, seqName, Nuc_or_AA):
    # the sequnece letters should be legal to rate4site
    if Nuc_or_AA == "AA":
        # AA seq
        if not re.match(r'^[ACDEFGHIKLMNPQRSTVWYBZXacdefghiklmnpqrstvwybzx]+$', aaSeq):
            return "illegal character was found in sequence: " + seqName
    else:
        # Nuc seq
        if not re.match(r'^[ACGTUINacgtuin]+$', aaSeq):
            return "illegal character was found in sequence: " + seqName
    return "yes" 


def check_if_no_overlap(max_overlap, ref_seq_details, s_bgn, s_end):
    ans = "check_if_no_overlap : no ans was picked"
    i = 0
    while i < len(ref_seq_details):
        fragment_beg = ref_seq_details[i][0]
        fragment_end = ref_seq_details[i][1]
        fragment_length = int(fragment_end) - int(fragment_beg) + 1
        if s_bgn <= fragment_beg and s_end >= fragment_end:
            # fragment is inside subjct
            return "previous fragment found %s_%s is fully inside new fragment" %(fragment_beg, fragment_end)
        elif s_bgn >= fragment_beg and s_end <= fragment_end:
            # subjct is inside fragment
            return "new fragment is fully inside previous fragment found " + str(fragment_beg + fragment_end)
        elif fragment_end < s_end and fragment_end > s_bgn:
            # fragment begins before subjct
            overlap_length = fragment_end - s_bgn + 1
            if overlap_length > fragment_length * max_overlap:

                return "overlap length of fragment is %d which is greater than maximum overlap: %d" %(overlap_length, fragment_length * max_overlap)
            else:
                # when the fragment might be a good match, we can only insert it if it did not match to all the fragments
                if i == len(ref_seq_details) - 1:
                    ans = "yes"
        elif fragment_beg > s_bgn and  fragment_beg < s_end:
            # fragment begins after subjct
            overlap_length = s_end - fragment_beg + 1
            if overlap_length > fragment_length * max_overlap:
                return "overlap length of fragment is %d which is greater than maximum overlap: %d" %(overlap_length, fragment_length * max_overlap)
            else:
                # when the fragment might be a good match, we can only insert it if it did not match to all the fragments
                if i == len(ref_seq_details) - 1:
                    ans = "yes"
        elif fragment_beg >= s_end or fragment_end <= s_bgn:
            # no overlap
            if i == len(ref_seq_details) - 1:
                ans = "yes"
        i += 1
    return ans


def count_letters(s):
    l = 0
    for c in s:
        if c.isalpha():
            l += 1
    return l

def cluster_homologoues(ref_cd_hit_hash, form, vars):

    msg = ""
    LOGGER.info("cluster_homologoues : create_cd_hit_output(%s, %s, %f, %s, %s);\n" %(vars['HITS_fasta_file'], vars['cd_hit_out_file'], vars['hit_redundancy']/100, ref_cd_hit_hash, form['DNA_AA']))
    total_num_of_hits = create_cd_hit_output(vars['HITS_fasta_file'], vars['cd_hit_out_file'], vars['hit_redundancy']/100, ref_cd_hit_hash, form['DNA_AA'])

    if form['MAX_NUM_HOMOL'].upper() == 'ALL':
        form['MAX_NUM_HOMOL'] = total_num_of_hits
    
    if total_num_of_hits < vars['min_num_of_hits']: # less seqs than the minimum: exit
        if total_num_of_hits <= 1:
            msg = "There is only 1 "
        else:
            msg = "There are only %d " %total_num_of_hits
        msg += "unique hits. The minimal number of sequences required for the calculation is %d. You may try to: " %vars['min_num_of_hits']
        msg += "Re-run the server with a multiple sequence alignment file of your own. Increase the Evalue. Decrease the Minimal %ID For Homologs"
        if int(form['ITERATIONS']) < 5:
            msg += " Increase the number of " + form['Homolog_search_algorithm'] + " iterations."
        msg += "\n"
        LOGGER.error("cluster_homologoues : " + msg)
    elif total_num_of_hits + 1 < vars['low_num_of_hits']: # less seqs than 10 : output a warning.
        msg = "Warning: There are "
        if total_num_of_hits + 1 < vars['number_of_homologoues_before_cd-hit']: # because we will add the query sequence itself to all the unique sequences.
            msg += "%d hits, only %d of them are" %(vars['number_of_homologoues_before_cd-hit'], total_num_of_hits+1)
        else:
            msg += str(total_num_of_hits + 1)
        msg += " unique sequences. The calculation is performed on the %d unique sequences, but it is recommended to run the program with a multiple sequence alignment file containing at least %s sequences." %(total_num_of_hits + 1, vars['low_num_of_hits'])
    else:
        msg = "There are %d %s hits. %d of them are unique, including the query. The calculation is performed on " %(vars['number_of_homologoues_before_cd-hit'], form['Homolog_search_algorithm'], total_num_of_hits + 1)
        if total_num_of_hits <= int(form['MAX_NUM_HOMOL']):
            msg += "%d unique sequences</a>." %(total_num_of_hits + 1)
        elif form['best_uniform_sequences'] == "best":
            msg += "the %s <a href=\"<?=$orig_path?>/%s\" style=\"color: #400080; text-decoration:underline;\">sequences</a> closest to the query (with the lowest E-value)." %(form['MAX_NUM_HOMOL'], vars['FINAL_sequences_html'])
        else:
            msg += "a sample of %s sequences that represent the list of homologues to the query." %form['MAX_NUM_HOMOL']
    LOGGER.info("cluster_homologoues : " + msg)

    #if os.path.exists(vars['HITS_rejected_file']) and os.path.getsize(vars['HITS_rejected_file']) != 0:
        #print_message_to_output("Here is the <a href=\"<?=$orig_path?>/" + vars['HITS_rejected_file'] + "\" TARGET=Rejected_Seqs style=\"color: #400080; text-decoration:underline;\">list of sequences</a> that produced significant alignments, but were not chosen as hits.")
        #print_message_to_output("<a href=\"<?=$orig_path?>/" + vars['HITS_rejected_file'] + "\" TARGET=Rejected_Seqs style=\"color: #400080; text-decoration:underline;\">Click here</a> if you wish to view the list of sequences which produced significant alignments, but were not chosen as hits.")
    return (total_num_of_hits + 1)

def create_cd_hit_output(input_file, output_file, cutoff, ref_cd_hit_hash, type):
    seq = ""
    seq_name = ""
    cmd = "" 
    n = 0

    # running cd-hit
    if type == "AA":
        cmd += "cd-hit -i %s -o %s " %(input_file, output_file)
        if cutoff > 0.7 and cutoff < 1:
            n = 5
        elif cutoff > 0.6 and cutoff <= 0.7:
            n = 4
        elif cutoff > 0.5 and cutoff <= 0.6:
            n = 3
        elif cutoff > 0.4 and cutoff <= 0.5:
            n = 2
    else:
        # DNA
        cmd += "cd-hit-est -i %s -o %s " %(input_file, output_file)
        if cutoff > 0.9 and cutoff < 1:
            n = 8
        elif cutoff > 0.88 and cutoff <= 0.9:
            n = 7
        elif cutoff > 0.85 and cutoff <= 0.88:
            n = 6
        elif cutoff > 0.8 and cutoff <= 0.85:
            n = 5
        elif cutoff > 0.75 and cutoff <= 0.8:
            n = 4
            
    cmd += "-c %f -n %d -d 0" %(cutoff, n)
    submit_job_to_Q("CD-HIT", cmd)
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        LOGGER.error(f"create_cd_hit_output : run of cd-hit fail. {output_file} is zero or not exists")
    num_cd_hits = 0
    try:
        CDHIT_OUTPUT = open(output_file, 'r')
    except:
        LOGGER.error(f"create_cd_hit_output: Can't open '{output_file}' for reading.\n")
        
    # inserting chosen homologues to a hash
    line = CDHIT_OUTPUT.readline()
    seq_name = ""
    seq = ""
    while line != "":
        line = line.strip()
        if line[0] == ">":
            seq_name = line[1:]
        else:
            seq = line
            if not seq_name in ref_cd_hit_hash:
                num_cd_hits += 1
                ref_cd_hit_hash[seq_name] =seq
        line = CDHIT_OUTPUT.readline()
    CDHIT_OUTPUT.close()	   
    return num_cd_hits

def add_sequences_removed_by_cd_hit_to_rejected_report(cd_hit_clusters_file, rejected_fragments_file, num_rejected_homologs, vars):

    LOGGER.info("add_sequences_removed_by_cd_hit_to_rejected_report : running add_sequences_removed_by_cd_hit_to_rejected_report(%s, %s, %d)\n" %(cd_hit_clusters_file, rejected_fragments_file, num_rejected_homologs))
    try:
        REJECTED = open(rejected_fragments_file, 'a')
    except:
        LOGGER.error(f"add_sequences_removed_by_cd_hit_to_rejected_report: Can't open '{rejected_fragments_file}' for writing.\n")
    try:
        CDHIT = open(cd_hit_clusters_file, 'r')
    except:
        LOGGER.error(f"add_sequences_removed_by_cd_hit_to_rejected_report: Can't open '{cd_hit_clusters_file}' for reading.\n")

    REJECTED.write("\n\t Sequences rejected in the clustering stage by CD-HIT\n\n")
    cluster_members = {}
    cluster_head = ""
    line = CDHIT.readline()
    while line != "":
        match = re.match(r'^>Cluster', line)
        if match:
            # New Cluster
            for cluster_member in cluster_members.keys():
                REJECTED.write("%d Fragment %s rejected: the sequence shares %s identity with %s (which was preserved)\n" %(num_rejected_homologs, cluster_member, cluster_members[cluster_member], cluster_head))
                num_rejected_homologs += 1
            cluster_members = {}
            cluster_head = ""
        else:
            # Clusters Members
            words = line.split()
            if len(words) > 2:
                x = words[2][1:-3] # delete the symbols > and ... from the beginning and and ending of the sequence name
                if words[3] == "*":
                    cluster_head  = x
                elif len(words) > 3:
                    cluster_members[x] = words[4]
        line = CDHIT.readline()
    vars['num_rejected_homologs'] = num_rejected_homologs

def choose_final_homologoues(ref_to_seqs_hash, ref_to_cd_hash, max_num_homologs, witch_unifrom, output_file, rejected_file, num_rejected_homologs, vars):

    LOGGER.info("sort_sequences_from_eval(%s ,%s , %f, %s, %s, %s, %d)\n" %("ref_to_seqs_hash", "ref_to_cd_hash", max_num_homologs, witch_unifrom, output_file, rejected_file, num_rejected_homologs))
    query_name = ""
    query_AAseq = ""
    counter = 1
    try:
        FINAL = open(vars['FINAL_sequences'], 'w')
    except:
        LOGGER.error(f"choose_final_homologoues : cannot open the file {vars['FINAL_sequences']} for writing.")

    # we write the query sequence to the file of the final homologs
    FINAL.write(">%s\n%s\n" %(vars['query_string'], vars['protein_seq_string']))
    final_file_size = os.path.getsize(vars['FINAL_sequences']) # take the size of the file before we add more sequences to it
    try:
        REJECTED = open(rejected_file, 'a')
    except:
        LOGGER.error(f"choose_final_homologoues : cannot open the file {rejected_file} for writing.")

    # write query details
    if query_AAseq != "":
        FINAL.write(">%s\n%s\n" %(query_name, query_AAseq))
    size_cd_hit_hash = len(ref_to_cd_hash)
    uniform = 1
    jump = 1
    if not witch_unifrom == "best":
        uniform = int(size_cd_hit_hash / max_num_homologs)
        if uniform == 0:
            uniform = 1
    final_number_of_homologoues = 1
    REJECTED.write("\n\tSequences rejected because of the requirement to select only %d representative homologs\n\n" %(max_num_homologs + 1))
    # write homologs
    for s_name in sorted(ref_to_seqs_hash.keys(), key = ref_to_seqs_hash.get):
        # write next homolog
        if s_name in ref_to_cd_hash: # and 'SEQ' in ref_to_cd_hash[s_name]:
            if counter != jump or counter > max_num_homologs * uniform:
                counter += 1
                REJECTED.write("%d %s\n" %(num_rejected_homologs, s_name))
                num_rejected_homologs += 1
                continue
            final_number_of_homologoues += 1  
            FINAL.write(">%s\n%s\n" %(s_name, ref_to_cd_hash[s_name]))
            counter += 1
            jump += uniform

    FINAL.close()
    REJECTED.close()
    vars['final_number_of_homologoues'] = final_number_of_homologoues
    # check that more sequences were added to the file
    if not final_file_size < os.path.getsize(vars['FINAL_sequences']):
        LOGGER.error(f"choose_final_homologoues : the file {vars['FINAL_sequences']} doesn't contain sequences")

def convert_rna_to_dna(Seqs, Seqs_dna):

    # replace the u with t and return the sequences names replaced

    try:

        OUT = open(Seqs_dna, 'w')

    except:

        return("convert_rna_to_dna: Can't open file " + Seqs_dna + " for writing.")

    try:

        SEQS = open(Seqs, 'r')

    except:

        return("convert_rna_to_dna: Can't open file " + Seqs + " for reading.")

    Seqs_Names = []
    seq_name = ""

    line = SEQS.readline()
    while line != "":

        line = line.rstrip()
        match1 = re.match(r'^>(.*)', line)
        if match1:

            seq_name = match1.group(1)


        elif 'u' in line or 'U' in line:

            Seqs_Names.append(seq_name)
            line = line.replace('u', 't')
            line = line.replace('U', 'T')

        OUT.write(line + "\n")
        line = SEQS.readline()

    OUT.close()
    SEQS.close()

    return("OK", Seqs_Names)

def make_sequences_file_HTML(plain_txt_sequences, HTML_sequences):
    try:
        HTML_SEQUENCES = open(HTML_sequences, 'w')
    except:
        LOGGER.error(f"make_sequences_file_HTML : cannot open the file {HTML_sequences} for writing.")
    try:
        TXT_SEQUENCES = open(plain_txt_sequences, 'r')
    except:
        LOGGER.error(f"make_sequences_file_HTML : cannot open the file {plain_txt_sequences} for reading.")
	
    counter = 1	
    line = TXT_SEQUENCES.readline()
    while line != "":
        line = line.strip()
        if line == "":
            line = TXT_SEQUENCES.readline()
            continue
        if line[0] != ">":
            counter += 1		
            HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3>" + line + "</FONT><BR>\n")
        else:
            line = line[1:]
            if line[:9] == "Input_seq":
                HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3>>%d_%s</FONT><BR>\n" %(counter, line))
            else:
                name = line.split("|")[1]
                if line[:3] == "ur|":
                    HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3><A HREF=\"https://www.uniprot.org/uniref/UniRef90_%s\">>%d_%s</A></FONT><BR>\n" %(name, counter, line))
                elif line[:3] == "up|":
                    HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3><A HREF=\"https://www.uniprot.org/uniprot/%s\">>%d_%s</A></FONT><BR>\n" %(name, counter, line))
                elif line[:3] == "gi|":
                    HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3><A HREF=\"https://www.ncbi.nlm.nih.gov/nuccore/%s\">>%d_%s</A></FONT><BR>\n" %(name, counter, line))
        line = TXT_SEQUENCES.readline()
    HTML_SEQUENCES.close()
    TXT_SEQUENCES.close()

def create_MSA(form, vars):

    if form['MSAprogram'] == "CLUSTALW":
        cmd = "clustalw -infile=%s -outfile=%s" %(vars['FINAL_sequences'], vars['msa_clustal'])
        LOGGER.info("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("clustalw", cmd)
        convert_msa_format(vars['msa_clustal'], "clustal", vars['msa_fasta'], "fasta")
    elif form['MSAprogram'] == "MAFFT":
        cmd = "mafft --localpair --maxiterate 1000 --quiet %s > %s" %(vars['FINAL_sequences'], vars['msa_fasta'])
        LOGGER.info("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("MAFFT", cmd)
        #convert_msa_format(vars['msa_fasta'], "fasta", vars['msa_clustal'], "clustal")
    elif form['MSAprogram'] == "PRANK":
        cmd = "%s/prank -d=%s -o=%s -F" %(GENERAL_CONSTANTS.PRANK, vars['FINAL_sequences'], vars['msa_fasta'])
        LOGGER.warn("PRANK is accurate but slow MSA program, please be patient.")
        LOGGER.info("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("PRANK", cmd)

        if os.path.exists(vars['msa_fasta'] + ".2.fas"):
            vars['msa_fasta'] += ".2.fas"
        elif os.path.exists(vars['msa_fasta'] + ".1.fas"):
            vars['msa_fasta'] += ".1.fas"
        elif os.path.exists(vars['msa_fasta'] + ".best.fas"):
            vars['msa_fasta'] +=  ".best.fas"
        #convert_msa_format(vars['msa_fasta'], "fasta", vars['msa_clustal'], "clustal")
    elif form['MSAprogram'] == "MUSCLE":
        cmd = "muscle -align %s -output %s" %(vars['FINAL_sequences'], vars['msa_fasta'])
        LOGGER.info("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("MUSCLE", cmd)
        #convert_msa_format(vars['msa_clustal'], "clustal", vars['msa_fasta'], "fasta")
    else:
        LOGGER.error(f"create_MSA : choose one of the programs for creating the msa: clustalw, mafft, prank or muscle.")

    if not os.path.exists(vars['msa_fasta']) or os.path.getsize(vars['msa_fasta']) == 0:
        LOGGER.error(f"create_MSA : the {form['MSAprogram']} program failed to create the MSA. Choose a different program to create the MSA.")

def convert_msa_format(infile, infileformat, outfile, outfileformat):

    try:
        AlignIO.convert(infile, infileformat, outfile, outfileformat)
    except:
        LOGGER.error(f"convert_msa_format : cannot convert {infile} from {infileformat} to {outfileformat}.")

def extract_data_from_MSA(form, vars):
    
    vars['query_string'] = form['msa_SEQNAME']
    vars['protein_seq_string'] = vars['MSA_query_seq']
    ## mode :  include msa and pdb
    if vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":
        compare_atom_seqres_or_msa("MSA", vars)

def find_best_substitution_model(form, vars):

    try:
        # convert fasta to phylip
        msa_phy_filepath = "input_msa.phy"
        #convert_msa_format(vars['msa_fasta'], "fasta", msa_phy_filepath, "phylip-relaxed")
        AlignIO.convert(vars['msa_fasta'], "fasta", msa_phy_filepath, "phylip-relaxed")
        os.chmod(vars['msa_fasta'], 0o644)
        os.chmod(msa_phy_filepath, 0o644)
        if form['DNA_AA'] == "Nuc":
            run_jmt(msa_phy_filepath, form, vars)
        else:
            run_prottest(msa_phy_filepath)
    except:
        vars['best_fit'] = "model_search_failed"
        form['SUB_MATRIX'] = "JTT"
        LOGGER.info("The evolutionary model search has failed. The JTT model is chosen by default.")


def run_jmt(msa_file_path, form, vars):

    JMT_JAR_FILE = GENERAL_CONSTANTS.JMODELTEST2
    output_file_path = "model_selection.txt"
    cmd = "java -Djava.awt.headless=true -jar %s -d %s -t BIONJ -AICc -f -o %s" %(JMT_JAR_FILE, msa_file_path, output_file_path)
    submit_job_to_Q("jmt", cmd)
    LOGGER.info("run_jmt: %s\n" %cmd)

    f = open(output_file_path, 'r')

    start_reading = False
    JMT_VALID_MODELS = ["JC","HKY","GTR"]

    # extract best model from table
    line = f.readline()
    while line != "":
        if start_reading:
            line = line.strip()
            split_row = line.split()
            model = split_row[0]
            if model in JMT_VALID_MODELS:
                f.close()
                #model = model.strip('()')
                if model == "JC":
                    form['SUB_MATRIX'] = "JC_Nuc"
                else:
                    form['SUB_MATRIX'] = model
                vars['best_fit'] = "model_found"
                LOGGER.info("The best evolutionary model was selected to be: " + model)
                return				
        elif re.search(r'Model             -lnL    K     AICc       delta       weight   cumWeight', line, re.M):
            start_reading = True
        line = f.readline()

    vars['best_fit'] = "model_search_failed"
    form['SUB_MATRIX'] = "JC_Nuc"
    LOGGER.info("The evolutionary model search has failed. The JC model is chosen by default")
    f.close()

def run_prottest(msa_file_path, form, vars):

    output_file_path = "model_selection.txt"
    cmd = "prottest -log disabled -i %s -AICC -o %s -S 1 -JTT -LG -MtREV -Dayhoff -WAG -CpREV -threads 1" %(msa_file_path, output_file_path)
    submit_job_to_Q("protest", cmd)
    LOGGER.info("run_protest: %s\n" %cmd)

    f = open(output_file_path, 'r')

    match = re.search(r"(?<=Best model according to AICc: ).*", f.read())
    f.close()
    if match:

        vars['best_fit'] = "model_found"
        model = match.group()
        model = model.strip('()')
        LOGGER.info("The best evolutionary model was selected to be: " + model)

    else:
	
        vars['best_fit'] = "model_search_failed"
        model = "JTT"
        LOGGER.info("The evolutionary model search has failed. The JTT model is chosen by default.")

    form['SUB_MATRIX'] = model


def run_rate4site_old(form, vars):
    rate4s = os.path.join(basedir, "rate4site_bioseq", "rate4site")
    rate4s_ML = os.path.join(basedir, "rate4site_bioseq", "rate4site.24Mar2010")
    rate4s_slow = os.path.join(basedir, "rate4site_bioseq", "rate4site.doubleRep")
    vars['r4s_log'] = "r4s.log" # log file
    vars['r4s_out'] = "r4s.res" # output file
    MatrixHash = {'JTT' : '-Mj', 'MTREV' : '-Mr', 'CPREV' : '-Mc', 'WAG' : '-Mw', 'DAYHOFF' : '-Md', 'T92' : '-Mt', 'HKY' : '-Mh', 'GTR' : '-Mg', 'JC_NUC' : '-Mn', 'JC_AA' : '-Ma', 'LG' : '-Ml'}

    params = " -a '%s' -s %s -zn %s -bn -o %s" %(vars['query_string'], vars['msa_fasta'], MatrixHash[(form['SUB_MATRIX']).upper()], vars['r4s_out'])
    if vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":
        params += " -t %s" %vars['tree_file']
		
    if form['ALGORITHM'] == "Bayes":
        params += " -ib -n 32 -v 9" 
        r4s_comm = rate4s + params
    else:
        params += " -im -v 9"
        r4s_comm = rate4s_ML + params 	

    r4s_comm += " -l " + vars['r4s_log']
    LOGGER.info("run_rate4site : running command: %s\n" %r4s_comm)
    submit_job_to_Q("rate4site", r4s_comm)
    # if the run failed - we rerun using the slow verion
    if check_if_rate4site_failed(vars['r4s_log'], vars):
        vars['r4s_log'] = "r4s_slow.log"
        LOGGER.info("run_rate4site : The run of rate4site failed. Sending warning message to output.\nThe same run will be done using the SLOW version of rate4site.\n")
        LOGGER.info("Warning: The given MSA is very large, therefore it will take longer for ConSurf calculation to finish. The results will be sent to the e-mail address provided.<br>The calculation continues nevertheless.")
        r4s_comm = rate4s_slow + params + " -l " + vars['r4s_log']
        LOGGER.info("run_rate4site : running command: %s\n" %r4s_comm)
        submit_job_to_Q("rate4siteSlow", r4s_comm)
        if check_if_rate4site_failed(vars['r4s_log']):
            LOGGER.error("run_rate4site : The run of rate4site failed. Sending warning message to output.\n")
    extract_diversity_matrix_info(vars['r4s_log'], vars)

def check_if_rate4site_failed(r4s_log, vars):
    res_flag = vars['r4s_out']
    if not os.path.exists(res_flag) or os.path.getsize(res_flag) == 0: # 1
        LOGGER.info(f"check_if_rate4site_failed : the file {res_flag} either does not exist or is empty. \n")
        return True

    try:
        R4S_RES = open(res_flag, 'r')
    except:
        LOGGER.info(f"check_if_rate4site_failed : can not open file: {res_flag}. aborting.\n")
        return True

    error = False
    line = R4S_RES.readline()
    while line != "":
        if "likelihood of pos was zero" in line:
            LOGGER.info(f"check_if_rate4site_failed : the line: \"likelihood of pos was zero\" was found in {r4s_log}.\n")
            error = True
            break
        if re.match(r'rate of pos\:\s\d\s=', line):
            break
        if "Bad format in tree file" in line:
            LOGGER.error(f"check_if_rate4site_failed : There is an error in the tree file format. Please check that your tree is in the <a href = \"{GENERAL_CONSTANTS.CONSURF_TREE_FAQ}\">requested format</a> and reupload it to the server.<br>\n")
        line = R4S_RES.readline()

    R4S_RES.close()
    return error


def extract_diversity_matrix_info(r4s_log_file, vars):

    # extracting diversity matrix info
    matrix_disINFO = "\"\""
    matrix_lowINFO = "\"\""
    matrix_upINFO = "\"\""

    try:
        RES_LOG = open(r4s_log_file, 'r')
    except:
        LOGGER.error(f"extract_diversity_matrix_info : can't open {r4s_log_file} for reading.\n")

    line = RES_LOG.readline()
    while line != "":
        line = line.rstrip()
        match1 = re.match(r'\#Average pairwise distance\s*=\s+(.+)', line)
        if match1:
            matrix_disINFO = match1.group(1)
        else:
            match2 = re.match(r'\#lower bound\s*=\s+(.+)', line)
            if match2:
                matrix_lowINFO = match2.group(1)
            else:
                match3 = re.match(r'\#upper bound\s*=\s+(.+)', line)
                if match3:
                    matrix_upINFO = match3.group(1)
                    break
        line = RES_LOG.readline()
    RES_LOG.close()
    vars['Average pairwise distance'] = matrix_disINFO

def assign_colors_according_to_r4s_layers(form, vars):

    LOGGER.info("assign_colors_according_to_r4s_layers(%s, %s)\n" %(vars['gradesPE_Output'], vars['r4s_out']))
    vars['insufficient_data'] = False
	# we extract the data from the rate4site output
    try:
        RATE4SITE = open(vars['r4s_out'], 'r')
    except:
        LOGGER.error(f"assign_colors_according_to_r4s_layers : can't open {vars['r4s_out']} for reading.\n")

    line = RATE4SITE.readline()
    while line != "":
        line.rstrip()
        if form['ALGORITHM'] == "Bayes":
            # baysean
            match1 = re.match(r'^\s*(\d+)\s+(\w)\s+(\S+)\s+\[\s*(\S+),\s*(\S+)\]\s+\S+\s+(\d+)\/(\d+)', line)
            if match1:
                vars['gradesPE_Output'].append({'POS' : int(match1.group(1)), 'SEQ' : match1.group(2), 'GRADE' : float(match1.group(3)), 'INTERVAL_LOW' : float(match1.group(4)), 'INTERVAL_HIGH' : float(match1.group(5)), 'MSA_NUM' : int(match1.group(6)), 'MSA_DENUM' : match1.group(7)})
        else:
            # Maximum likelihood
            match2 = re.match(r'^\s*(\d+)\s+(\w)\s+(\S+)\s+(\d+)\/(\d+)', line)
            if match2:
                vars['gradesPE_Output'].append({'POS' : int(match2.group(1)), 'SEQ' : match2.group(2), 'GRADE' : float(match2.group(3)), 'INTERVAL_LOW' : float(match2.group(3)), 'INTERVAL_HIGH' : float(match2.group(3)), 'MSA_NUM' : int(match2.group(4)), 'MSA_DENUM' : match2.group(5)})
        line = RATE4SITE.readline()
    RATE4SITE.close()
    # we find the maximum and the minimum scores    
    max_cons = vars['gradesPE_Output'][0]['GRADE']
    min_cons = vars['gradesPE_Output'][0]['GRADE']
    for element in vars['gradesPE_Output']:
        if element['GRADE'] < max_cons:
            max_cons = element['GRADE']
        if element['GRADE'] > min_cons:
            min_cons = element['GRADE']

    # we divide the interval between min_cons to max_cons to nine intervals
    # 4 intervals on the left side are of length 2 * min_cons / 9
    # 4 intervals on the right side are of length 2 * max_cons / 9
    NoLayers = 10
    LeftLayers = 5
    RightLayers = 5
    ColorLayers = []
    i = 0
    while i < LeftLayers:
        ColorLayers.append(max_cons * ((9 - 2 * i) / 9.0))
        i += 1
        
    i = 0
    while i < RightLayers:
        ColorLayers.append(min_cons * ((1 + 2 * i) / 9.0))
        i += 1

    # each position gets a grade according to the layer its score is in
    for element in vars['gradesPE_Output']:
        i = 0
        while not 'INTERVAL_LOW_COLOR' in element or not 'INTERVAL_HIGH_COLOR' in element or not 'COLOR' in element:
            if not 'INTERVAL_LOW_COLOR' in element:
                if i == NoLayers - 1:
                    element['INTERVAL_LOW_COLOR'] = 8
                elif element['INTERVAL_LOW'] >= ColorLayers[i] and element['INTERVAL_LOW'] < ColorLayers[i + 1]:
                    element['INTERVAL_LOW_COLOR'] = i
                elif element['INTERVAL_LOW'] < ColorLayers[0]:
                    element['INTERVAL_LOW_COLOR'] = 0

            if not 'INTERVAL_HIGH_COLOR' in element:
                if i == NoLayers - 1:
                    element['INTERVAL_HIGH_COLOR'] = 8
                elif element['INTERVAL_HIGH'] >= ColorLayers[i] and element['INTERVAL_HIGH'] < ColorLayers[i + 1]:
                    element['INTERVAL_HIGH_COLOR'] = i
                elif element['INTERVAL_HIGH'] < ColorLayers[0]:
                    element['INTERVAL_HIGH_COLOR'] = 0

            if not 'COLOR' in element:
                if i == NoLayers - 1:
                    element['COLOR'] = ColorScale[i - 1]
                elif element['GRADE'] >= ColorLayers[i] and element['GRADE'] < ColorLayers[i + 1]:
                    element['COLOR'] = ColorScale[i]
            i += 1

		# there is insufficient data if there are more than 3 layers in the confidence interval or the number of homologs in the MSA where the position is not empty is less than 5
        if element['INTERVAL_HIGH_COLOR'] - element['INTERVAL_LOW_COLOR'] > bayesInterval or element['MSA_NUM'] <= 5:
            element['ISD'] = 1
            vars['insufficient_data'] = True
        else:
            element['ISD'] = 0
    vars['layers_array'] = ColorLayers
    LOGGER.info("assign_colors_according_to_r4s_layers : color layers are %s\n" %str(vars['layers_array']))

def write_MSA_percentage_file(form, vars):
    
    # writes a file with the precentage of each acid in each position 
    #index_of_pos = get_positions_in_MSA() 
    percentage_per_pos = [] # precentage of each acid in each position in the MSA
    #unknown_per_pos = [] # precentage of unknown acid in each position in the MSA
    number_of_positions = len(vars['protein_seq_string'])
    #number_of_positions = len(index_of_pos)
    query_with_gaps = get_query_seq_in_MSA(vars) # query sequence with gaps

    if form['DNA_AA'] == "AA":
        acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
        unknown = "X"
        for i in range(number_of_positions):
            percentage_per_pos.append({})
            #unknown_per_pos.append(0)
    else:
        acids = ["A", "C", "G", "T", "U", "N"]
        unknown = "N"
        for i in range(number_of_positions):
            percentage_per_pos.append({})
            #unknown_per_pos.append(0)
    try:
        FASTA = open(vars['msa_fasta'], 'r')
    except:
        LOGGER.error("write_MSA_percentage_file : can't open the file %s for reading.\n" %vars['msa_fasta'])
    
    try:
        PRECENTAGE_FILE = open(vars['Msa_percentageFILE'], 'w')
    except:
        LOGGER.error("write_MSA_percentage_file : can't open the file %s for writing.\n" %vars['Msa_percentageFILE'])
    
    # we find the precentage of each acid in each position 
    seq = ""
    first = True
    line = FASTA.readline()
    while True:
        if line == "" or line[0] == '>':
            if first:
                first = False 
            else:
                #pos = 0
                #for i in index_of_pos:
                i = 0
                for pos in range(number_of_positions):
                    while i < len(query_with_gaps) and query_with_gaps[i] == '-':
                        i += 1
                    char = seq[i]
                    if char in  acids:
                        if char in percentage_per_pos[pos]:
                            percentage_per_pos[pos][char] += 1
                        else:
                            percentage_per_pos[pos][char] = 1
                    """
                    elif char != '-':
                        unknown_per_pos[pos] += 1
                    """
                    i += 1                       
                seq = ""
            # this is for the last sequence in the msa
            if line == "":
                break
        else:
            seq += line.strip().upper()
        line = FASTA.readline()   
    FASTA.close()
    
    # sort dictionaries
    for pos in range(number_of_positions):
        # we sort the amino acids but not the unknown character 
        unknown_percent = percentage_per_pos[pos].pop(unknown, None)
        percentage_per_pos[pos] = dict(sorted(percentage_per_pos[pos].items(), key=lambda item: item[1], reverse=True))
        if unknown_percent:
            percentage_per_pos[pos][unknown] = unknown_percent
		
    # calculate the percentage 
    for pos in range(number_of_positions):
        sum = 0.0
        for char in percentage_per_pos[pos]:
            sum += percentage_per_pos[pos][char]
        #sum += unknown_per_pos[pos]
        for char in percentage_per_pos[pos]:
            percentage_per_pos[pos][char] = 100 * (percentage_per_pos[pos][char] / sum)
            
        #unknown_per_pos[pos] = 100 * (unknown_per_pos[pos] / sum)
            
    # we write the file
    PRECENTAGE_FILE.write("\"The table details the residue variety in % for each position in the query sequence.\"\n\"Each column shows the % for that amino-acid, found in position ('pos') in the MSA.\"\n\"In case there are residues which are not a standard amino-acid in the MSA, they are represented under column 'OTHER'\"\n\npos,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,OTHER,MAX ACID,ConSurf grade\n\n")   
    pos = 0
    for i in range(number_of_positions):
        if vars['protein_seq_string'][i] == unknown:
            continue
		# position
        PRECENTAGE_FILE.write("%d" %(pos + 1))
		# known acids
        for char in acids:
            PRECENTAGE_FILE.write(",")
            if char in percentage_per_pos[pos]:
                PRECENTAGE_FILE.write("%.3f" %percentage_per_pos[pos][char])
            else:
                PRECENTAGE_FILE.write("0")
                
		# unknown acids
        #PRECENTAGE_FILE.write(",%.3f" %unknown_per_pos[pos])
		# max acid
        keys = list(percentage_per_pos[pos].keys())
        max_acid = ",%s %.3f" %(keys[0], percentage_per_pos[pos][keys[0]])
        PRECENTAGE_FILE.write(max_acid.rstrip("0").rstrip("."))
        # ConSurf grade
        PRECENTAGE_FILE.write(",%s\n" %vars['gradesPE_Output'][pos]['COLOR'])
        pos += 1
		
    PRECENTAGE_FILE.close()
    vars['percentage_per_pos'] = percentage_per_pos
    #vars['unknown_per_pos'] = unknown_per_pos
    if form['DNA_AA'] != "AA": 
        vars['B/E'] = False
        return
		
    vars['B/E'] = True
    buried_exposed = []
    reveal_buried_exposed(buried_exposed, vars)
    pos = 0 
    for element in vars['gradesPE_Output']:
        while pos < number_of_positions and vars['protein_seq_string'][pos] == unknown:
            pos += 1
        element['B/E'] = buried_exposed[pos]
        pos += 1
        if element['B/E'] == "e":
            if element['COLOR'] == 9 or element['COLOR'] == 8:
                element['F/S'] = "f"
            else:
                element['F/S'] = " "
        elif element['COLOR'] == 9:
            element['F/S'] = "s"
        else:
            element['F/S'] = " "

def get_query_seq_in_MSA(vars):
    # returns the query sequence with gaps as it's in the MSA
    try:
        FASTA = open(vars['msa_fasta'], 'r')
    except:
        LOGGER.error("get_query_seq_in_MSA : can't open the file %s for reading.\n" % vars['msa_fasta'])
    
    found = False
    seq = ""
    line = FASTA.readline()
    while line != "":
        first_word = line.split()[0]
        if found:
            if first_word[0] == '>':
                break
            else:
                seq += first_word
        elif first_word == ">"  + vars['msa_SEQNAME']:
                found = True
        line = FASTA.readline()   
    FASTA.close()
    return seq

def reveal_buried_exposed(buried_exposed, vars):
    
    # this function reveals the buried and exposed atoms.
    # We look at 6 positions before and 6 position after the current position, 260 numbers in total. 
    # We multiply each number by a unique weight and then we sum the numbers. 
    # The results to put in the formula 1 / (1 + e^(-x)). 
    # We do this twenty times with different weights. Now we have 20 numbers. 
    # We again multiply the numbers by different weights and sum them up and put them in the formula  1 / (1 + e^(-x)). 
    # We do this twice, again with different weights. Now we have two numbers. 
    # If number 1 > number2 we say the position is exposed, if not it’s buried.
    
    num_pos = len(vars['percentage_per_pos'])
    print_pnet_file(num_pos, vars)
    G_nl = 2
    G_N = [260, 20, 2]
    
    WEIGHT_BIN = os.path.join(basedir, "WEIGHT.BIN")
    weights_file = open(WEIGHT_BIN, 'r')
    pnet_file = open("p.net", 'r')
    
    for i in range(num_pos):
        G_o = [[], [],[]]
        for j in range(G_N[0]):
            G_o[0].append(read_number(pnet_file) / 100.0)
        for s in range(1, G_nl + 1):
            for k in range(G_N[s]):
                weight = read_number(weights_file)
                for j in range(G_N[s - 1]):
                    weight += read_number(weights_file) * G_o[s - 1][j]
                G_o[s].append(1 / (1 + math.exp(-weight)))

        weights_file.seek(0) # each time we use the same weights, so we move the pointer to the start
        if G_o[G_nl][0] < G_o[G_nl][1]:
            buried_exposed.append("e")
        else:
            buried_exposed.append("b")
            
    weights_file.close()
    pnet_file.close()
    return buried_exposed

def print_pnet_file(num_pos, vars):
    
    # for each position in the msa we print percentage per positions for 6 positions before and after
    window = 6
    pad = []
    pnet_file = open("p.net", 'w')
    acids = ["V", "L", "I", "M", "F", "W", "Y", "G", "A", "P", "S", "T", "C", "H", "R", "K", "Q", "E", "N", "D"]
    for i in range(window):
        pad.append({})
        for acid in acids:
            pad[-1][acid] = 0
            
    padded_percentage_per_pos = pad + vars['percentage_per_pos'] + pad
    for i in range(window, num_pos + window):
        j = -window
        while j < window + 1:
            for acid  in acids:
                if acid in padded_percentage_per_pos[i + j]:
                    pnet_file.write(str(int(padded_percentage_per_pos[i + j][acid])) + " ")
                else:
                    pnet_file.write("0 ")
            j += 1
        pnet_file.write("\n")
    pnet_file.close() 

def read_number(file):
    # reads numbers from a numbers file without storing the whole file in a string
    number = ""
    char = file.read(1)
    while char and char.isspace(): # first we skip white spaces
        char = file.read(1)
    while char and not char.isspace(): # we read the number
        number += char
        char = file.read(1)
    if number != "":
        number = float(number)
    return number


def consurf_create_output(form, vars):
    r4s2pdb = {} # key: poistion in SEQRES/MSA, value: residue name with position in atom (i.e: ALA:22:A)
    if vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree" or vars['SEQRES_seq'] != "":
        match_pdb_to_seq(r4s2pdb, vars['seqres_or_msa_seq_with_gaps'], vars['ATOM_seq_with_gaps'], vars['pdb_object'], form)
    else: # no seqres and no msa
        find_pdb_position(r4s2pdb, vars['pdb_object'], form, vars) 

    identical_chains = find_identical_chains_in_PDB_file(vars['pdb_object'], form['PDB_chain'], form, vars)
    atom_grades = {}
    create_gradesPE(vars['gradesPE'], r4s2pdb, vars['pdb_file_name'], form['PDB_chain'], vars['Used_PDB_Name'], vars['pdb_object'], identical_chains, vars['cif_or_pdb'], atom_grades, form, vars)
    replace_TmpFactor_Consurf_Scores(atom_grades, form['PDB_chain'], vars['pdb_file_name'], vars['Used_PDB_Name'], form, vars) # Create ATOMS section and replace the TempFactor Column with the ConSurf Grades (will create also isd file if relevant)
    # create_pdf(vars)

def match_pdb_to_seq(ref_fas2pdb, query_seq, pdbseq, pdb_object, form):

    UnKnownChar = ""
    if form['DNA_AA'] == "AA":
        UnKnownChar = "X"
    else:
        UnKnownChar = "N"

    # creating the hash that matches the position in the ATOM fasta to its position
    # in the pdb file and also the fasta ATOM position to the correct residue
    match_ATOM = pdb_object.get_positions()
    query_pos = 1
    pdb_pos = 1
    for pos in range(len(query_seq)):
        if query_seq[pos] != '-' and query_seq[pos] != UnKnownChar:
            if pdbseq[pos] == '-' or pdb_pos not in match_ATOM: 
                ref_fas2pdb[query_pos] = '-'
            else:
                ref_fas2pdb[query_pos] = match_ATOM[pdb_pos]
                pdb_pos += 1
            query_pos += 1
        elif pdbseq[pos] != '-':
            pdb_pos += 1
            
def find_pdb_position(ref_fas2pdb, pdb_object, form, vars):

    # finds the position of the the sequence in the pdb

    LOGGER.info("find_pdb_position(ref_fas2pdb, pdb_object)\n")
    UnKnownChar = ""
    if form['DNA_AA'] == "AA":
        UnKnownChar = "X"
    else:
        UnKnownChar = "N"
	
    match_ATOM = pdb_object.get_positions()
    # rate4site deletes the unknown chars
    rate4site_pos = 1
    pdb_pos = 1
    for char in vars['ATOM_without_X_seq']:
        if char != UnKnownChar:
            ref_fas2pdb[rate4site_pos] = match_ATOM[pdb_pos]                
            rate4site_pos += 1
        pdb_pos += 1

def find_identical_chains_in_PDB_file(pdb_Object, query_chain, form, vars):

    # Looking in the PDB for chains identical to the original chain
    ATOMS = pdb_Object.get_ATOM_withoutX()
    # string with identical chains
    identicalChains = query_chain

    # looking for chains identical to the original chain
    for chain in ATOMS:
        if query_chain != chain:
            other_seq = ATOMS[chain]
            query_seq = ATOMS[query_chain]
            chain_length = len(other_seq)
            OrgChain_length = len(query_seq)

            # if length not similar, skip
            if min(OrgChain_length, chain_length)/max(OrgChain_length, chain_length) <= 0.9:
                continue
            # compare the two chains 
            try:
                if pairwise_alignment(other_seq, query_seq, form=form, vars=vars) > 0.95:
                    identicalChains += " " + chain
            except:
                LOGGER.info("find_identical_chains_in_PDB_file: Error comparing the chains %s and %s\n" %(query_chain, chain))
    return identicalChains


def pairwise_alignment(first_seq, second_seq, clustalw_aln = "", seq_type = "", form=None, vars=None):
	# for new Bio
    aligner = Align.PairwiseAligner()
    # Pairwise Alignment Paramaters
    aligner.mode = 'global' #Can be either global or local, if undetermined, biopython will choose optimal algorithem

    if form['DNA_AA'] == "AA":
        aligner.substitution_matrix = substitution_matrices.load(os.path.join(basedir, "matrix.txt"))
    else:
        aligner.substitution_matrix = substitution_matrices.load(os.path.join(basedir, "matrix-nuc.txt"))

    #Default Gap extension and opening penalties for ClustalW are 0.2 and 10.0.
    aligner.open_gap_score = -5.0 #-10.0
    aligner.extend_gap_score = -0.20
    #aligner.target_end_gap_score = 0.0
    #aligner.query_end_gap_score = 0.0
	
    alignments = aligner.align(first_seq, second_seq)
    #[first_seq_with_gaps, middle_line, second_seq_with_gaps] = (str(alignments[0])).split() # old Bio
    first_seq_with_gaps = ""
    second_seq_with_gaps = ""
    alignment_string = str(alignments[0])
    lines = alignment_string.split('\n')
    # biopython 1.85
    # for line in lines:
    #     words = line.split()
        # if len(words) > 2:
        #     if words[0] == "target":
        #         first_seq_with_gaps += words[2]
        #     elif words[0] == "query":
        #         second_seq_with_gaps += words[2]              
    # biopython 1.79
    first_seq_with_gaps = lines[0]
    second_seq_with_gaps = lines[2]

	
    matches = 0
    length_without_gaps = 0
    for i in range(len(first_seq_with_gaps)):
        if first_seq_with_gaps[i] != '-' and second_seq_with_gaps[i] != '-':
            length_without_gaps += 1
            if first_seq_with_gaps[i] == second_seq_with_gaps[i]:
                matches += 1
    identity = (matches * 100.0) / length_without_gaps
	
    if clustalw_aln == "":
        # we don't write the alignment
        return identity
    try:
        CLUSTALW_ALN = open(clustalw_aln, 'w')
    except:
        LOGGER.error(f"write_alignment : could not open {clustalw_aln} for writing.\n")

    if seq_type == "SEQRES":
        CLUSTALW_ALN.write("target - Seqres sequence\nguery - Atom sequence\n\n" + alignment_string)
    else:
        CLUSTALW_ALN.write("target - MSA sequence\nguery - Atom sequence\n\n" + alignment_string)

    CLUSTALW_ALN.close()
    return (first_seq_with_gaps, second_seq_with_gaps, identity)


def create_gradesPE(gradesPE, ref_match = "", pdb_file = "", chain = "", prefix = "", pdb_object = "", identical_chains = "", pdb_cif = "", atom_grades = "", form=None, vars=None):
    # printing the the ConSurf gradesPE file

    if pdb_cif == "pdb": # this is for the pipe file
	
        seq3d_grades_isd = ""
        seq3d_grades = ""
        # arrays showing how to color the residues. The subarrays holds residues of the same color
        no_isd_residue_color = [[],[],[],[],[],[],[],[],[],[],[]] # no insufficient data
        isd_residue_color = [[],[],[],[],[],[],[],[],[],[],[]] # insufficient data

    try:
        PE = open(gradesPE, 'w')
    except:
        LOGGER.error( "create_gradesPE : can't open '" + gradesPE + "' for writing.")
    if form['DNA_AA'] == "AA":
        unknown_char = "X"
        PE.write("\t Amino Acid Conservation Scores\n")
    else:
        unknown_char = "N"
        PE.write("\t Nucleic Acid Conservation Scores\n")

    PE.write("\t=======================================\n\n")
    PE.write("The layers for assigning grades are as follows.\n")
    for i in range(1, len(vars['layers_array'])):
        if vars['layers_array'][i - 1] < 0:
            left_end = "%.3f" %vars['layers_array'][i - 1]
        else:
            left_end = " %.3f" %vars['layers_array'][i - 1]
        if vars['layers_array'][i] < 0:
            right_end = "%.3f" %vars['layers_array'][i]
        else:
            right_end = " %.3f" %vars['layers_array'][i]
        PE.write("from %s to %s the grade is %d\n" %(left_end, right_end, 10 - i))
    PE.write("\nIf the difference between the colors of the CONFIDENCE INTERVAL COLORS is more than 3 or the msa number (under the column titled MSA) is less than 6, there is insufficient data and an * appears in the COLOR column.\n")
    PE.write("\n- POS: The position of the acid in the sequence.\n")
    PE.write("- SEQ: The acid one letter.\n")
    PE.write("- ATOM: When there's a model, The ATOM derived sequence in three letter code, including the acid's positions as they appear in the PDB file and the chain identifier.\n")
    PE.write("- SCORE: The normalized conservation scores.\n")
    PE.write("- COLOR: The color scale representing the conservation scores (9 - conserved, 1 - variable).\n")
    PE.write("- CONFIDENCE INTERVAL: When using the bayesian method for calculating rates, a confidence interval is assigned to each of the inferred evolutionary conservation scores, next to it are the colors of the lower and upper bounds of the confidence interval\n")
    if vars['B/E']:
        PE.write("- B/E: Burried (b) or Exposed (e) residue.\n")
        PE.write("- F/S: functional (f) or structural (s) residue (f - highly conserved and exposed, s - highly conserved and burried).\n")
    PE.write("- MSA DATA: The number of aligned sequences having an acid (non-gapped) from the overall number of sequences at each position.\n")
    PE.write("- RESIDUE VARIETY: The residues variety at each position of the multiple sequence alignment.\n\n")

    if form['ALGORITHM'] == "Bayes":
        CONFIDENCE_INTERVAL = "CONFIDENCE INTERVAL\t"
    else:
        CONFIDENCE_INTERVAL = ""
  
    # the size of the POS, ATOM and MSA DATA columns is variable 
    pos_number_size = len(str(len(vars['protein_seq_string']))) # size of the of the number of positions
    pos_column_title_size = len("POS") # size of the title of the column POS
    pos_column_size = max(pos_column_title_size, pos_number_size) # size of the column POS
    num_spaces = pos_column_size - pos_column_title_size # number of spaces to add 
    while num_spaces > 0:
        PE.write(" ")
        num_spaces -= 1
		
    PE.write("POS\tSEQ\t")
    if ref_match != "":
        # consurf. in conseq there is no atom because there is no model
        max_res_details = pdb_object.get_max_res_details()
        atom_column_title_size = len("ATOM") # size of the title of the column ATOM
        atom_column_size = max(atom_column_title_size, max_res_details) # size of the column ATOM
        num_spaces = atom_column_size - atom_column_title_size # number of spaces to add

        while num_spaces > 0:
            PE.write(" ")
            num_spaces -= 1
        PE.write("ATOM\t")
		
    if vars['B/E']:
        PE.write(" SCORE\tCOLOR\t%sB/E\tF/S\t" %CONFIDENCE_INTERVAL)
    else:
        PE.write(" SCORE\tCOLOR\t%s" %CONFIDENCE_INTERVAL)
                
    msa_size = len(str(vars['final_number_of_homologoues'])) # size of the number of homologs
    msa_column_title_size = len("MSA DATA") # size of the title of the column MSA DATA
    msa_column_size = max(msa_column_title_size, 2 * msa_size + 1) # size of the column MSA DATA
    num_spaces = msa_column_size - msa_column_title_size # number of spaces to add 
    while num_spaces > 0:
        PE.write(" ")
        num_spaces -= 1
		
    PE.write("MSA DATA\tRESIDUE VARIETY\n\n")
    seq_index = 0 # the index of the position in the query sequence (the rate4site output doesn't contain unknown chars)		
    for elem in vars['gradesPE_Output']:
        pos = elem['POS']
        var = ""
        num_spaces = pos_column_size - len(str(pos)) # number of spaces to add 

        while num_spaces > 0:
            PE.write(" ")
            num_spaces -= 1
			
        PE.write("%d\t" %pos)
        PE.write("  %s\t" %elem['SEQ'])
        if ref_match != "":
            # consurf, in conseq there is no atom because there is no model
            atom_3L = ref_match[pos]
            num_spaces = atom_column_size - len(atom_3L) # number of spaces to add
            while num_spaces > 0:
                PE.write(" ")
                num_spaces -= 1
            PE.write("%s\t" %atom_3L)
            # save the grade of the residue inorder to write it on the pdb file
            if atom_3L != '-':
                residue_number = atom_3L.split(':')[1]
                atom_grades[residue_number] = [str(elem['COLOR']), elem['ISD'], "%6.3f" %elem['GRADE']]

        PE.write("%6.3f\t" %elem['GRADE'])
        if elem['ISD'] == 1:
            PE.write("   %d*\t" %elem['COLOR'])
        else:
            PE.write("   %d \t" %elem['COLOR'])
        if form['ALGORITHM'] == "Bayes":
            PE.write("%6.3f, " %elem['INTERVAL_LOW'])
            PE.write("%6.3f  " %elem['INTERVAL_HIGH'])
            PE.write("%d," %ColorScale[elem['INTERVAL_LOW_COLOR']])
            PE.write("%d\t" %ColorScale[elem['INTERVAL_HIGH_COLOR']])
        if vars['B/E']:
            PE.write("  %s\t  %s\t" %(elem['B/E'], elem['F/S']))
        homologs_in_pos = str(elem['MSA_NUM']) + "/" + elem['MSA_DENUM'] # number of homologs in the position
        num_spaces = msa_column_size - len(homologs_in_pos) # number of spaces to add 

        while num_spaces > 0:
            PE.write(" ")
            num_spaces -= 1
        PE.write(homologs_in_pos)
		
		# we write the acids percentage in the msa
        while seq_index < len(vars['protein_seq_string']) and vars['protein_seq_string'][seq_index] == unknown_char:
            seq_index += 1
        for char in vars['percentage_per_pos'][seq_index]:
            if vars['percentage_per_pos'][seq_index][char] == 100:
                var += char
            else:
				# if the precentage is less than one write <1%
                if vars['percentage_per_pos'][seq_index][char] > 1:
                    var += "%s %2d%%, " %(char, vars['percentage_per_pos'][seq_index][char])
                else:
                    var += "%s <1%%, " %char

        if len(vars['percentage_per_pos'][seq_index]) != 1:
            var = var[:-2] # we delete the last comma
        PE.write("\t" + var + "\n")
        seq_index += 1

        # the amino-acid in that position, must be part of the residue variety in this column
        if not re.search(elem['SEQ'], var, re.IGNORECASE):
            PE.close()
            LOGGER.error("create_gradesPE : in position %s, the amino-acid %s does not match the residue variety: %s." %(pos, elem['SEQ'], var))

        if pdb_cif != "pdb": # the next part is for the pipe file
            continue

        # printing the residue to the rasmol script
        # assigning grades to seq3d strings
        if not '-' in atom_3L:
            atom_3L = re.search(r'(.+):', atom_3L).group(1)
            if form['DNA_AA'] == "Nuc":

                atom_3L = "D" + atom_3L

            color = elem['COLOR']
            no_isd_residue_color[color].append(atom_3L)
            if elem['ISD'] == 1:

                isd_residue_color[10].append(atom_3L)
                seq3d_grades_isd += "0"

            else:

                isd_residue_color[color].append(atom_3L)
                seq3d_grades_isd += str(color)

            seq3d_grades += str(color)

        else:

            seq3d_grades_isd += "."
            seq3d_grades += "."

    PE.write("\n\n*Below the confidence cut-off - The calculations for this site were performed on less than 6 non-gaped homologue sequences,\n")
    PE.write("or the confidence interval for the estimated score is equal to- or larger than- 4 color grades.\n")
    PE.close()

    if pdb_cif != "pdb": # the next part is for the pipe file
	
        return

    if seq3d_grades_isd == "" or seq3d_grades == "":

        LOGGER.error("create_gradesPE : there is no data in the returned values seq3d_grades_isd or seq3d_grades from the routine")

    # This will create the pipe file for FGiJ
    pipeFile = prefix + "_consurf_firstglance.pdb"
    pipeFile_CBS = prefix + "_consurf_firstglance_CBS.pdb" # pipe for color blind friendly
    create_pipe_file(pipeFile, pipeFile_CBS, seq3d_grades, seq3d_grades_isd, isd_residue_color, no_isd_residue_color, pdb_file, chain, (prefix).upper(), identical_chains, pdb_object, form, vars)

    # create RasMol files
    create_rasmol(prefix, chain, no_isd_residue_color, isd_residue_color, vars)

def create_pipe_file(pipeFile, pipeFile_CBS, seq3d_grades, seq3d_grades_isd, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, pdb_file_name, user_chain, IN_pdb_id_capital, identical_chains, pdb_object,
                     form, vars):

    # CREATE PART of PIPE
    partOfPipe = "partOfPipe"
    partOfPipe_CBS = "partOfPipe_CBS"
	
    length_of_seqres = pdb_object.get_num_known_seqs()
    length_of_atom = pdb_object.get_num_known_atoms()

    LOGGER.info("create_part_of_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(partOfPipe, vars['unique_seqs'], "user_DB", "seq3d_grades_isd", "seq3d_grades", length_of_seqres, length_of_atom, "isd_residue_color_ArrRef", "no_isd_residue_color_ArrRef", form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], "legacy"))
    create_part_of_pipe_new(partOfPipe, vars['unique_seqs'], "user_DB", seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], vars['Average pairwise distance'], "legacy")
					  																									 																															

    # create the color blind friendly version
    create_part_of_pipe_new(partOfPipe_CBS, vars['unique_seqs'], "user_DB", seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], vars['Average pairwise distance'], "cb")
																																

    LOGGER.info("extract_data_from_pdb(%s)\n" %pdb_file_name)
    header_pipe = extract_data_from_pdb(pdb_file_name)
										   

    # GET THE FILE NAMES
    msa_filename = ""
    msa_query_seq_name = ""
    if vars['user_msa_file_name'] is not None:

        msa_filename = vars['user_msa_file_name']
        msa_query_seq_name = form['msa_SEQNAME']

    tree_filename = ""
    if form['tree_name'] is not None:

        tree_filename = vars['tree_file']

    # GET THE CURRENT TIME
    completion_time = str(datetime.now().time())
    run_date = str(datetime.now().date())

    # USE THE CREATED PART of PIPE to CREATE ALL THE PIPE TILL THE PDB ATOMS (DELETE THE PART PIPE)
    LOGGER.info("create_consurf_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(vars['working_dir'], IN_pdb_id_capital, user_chain, "header_pipe", pipeFile, identical_chains, partOfPipe, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date))
    create_consurf_pipe_new(vars['working_dir'], IN_pdb_id_capital, user_chain, header_pipe, pipeFile, identical_chains, partOfPipe, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date)
				   																								
    # USE THE CREATED PART of PIPE to CREATE ALL THE PIPE TILL THE PDB ATOMS (DELETE THE PART PIPE) - Color friendly version
    LOGGER.info("create_consurf_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(vars['working_dir'], IN_pdb_id_capital, user_chain, "header_pipe", pipeFile_CBS, identical_chains, partOfPipe_CBS, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date))
    create_consurf_pipe_new(vars['working_dir'], IN_pdb_id_capital, user_chain, header_pipe, pipeFile_CBS, identical_chains, partOfPipe_CBS, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date)
				   
    # Add the PDB data to the pipe
    LOGGER.info("add_pdb_data_to_pipe(%s, %s)\n" %(pdb_file_name, pipeFile))
    add_pdb_data_to_pipe(pdb_file_name, pipeFile)

    # Add the PDB data to the pipe - color blind version
    LOGGER.info("add_pdb_data_to_pipe(%s, %s)\n" %(pdb_file_name, pipeFile_CBS))
    add_pdb_data_to_pipe(pdb_file_name, pipeFile_CBS)


def create_part_of_pipe_new(pipe_file, unique_seqs, db, seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, ref_isd_residue_color, ref_no_isd_residue_color, E_score, iterations, max_num_homol, MSAprogram, algorithm, matrix, Average_pairwise_distance, scale = "legacy"):

    # creating part of the pipe file, which contains all the non-unique information.
    # each chain will use this file to construct the final pdb_pipe file, to be viewed with FGiJ

    if scale == "legacy":

        scale_block = "!color color_grade0 FFFF96 insufficient data yellow\n!color color_grade1 10C8D1 turquoise variable\n!color color_grade2 8CFFFF\n!color color_grade3 D7FFFF\n!color color_grade4 EAFFFF\n!color color_grade5 FFFFFF\n!color color_grade6 FCEDF4\n!color color_grade7 FAC9DE\n!color color_grade8 F07DAB\n!color color_grade9 A02560 burgundy conserved"

    else:

        scale_block = "!color color_grade0 FFFF96 insufficient data yellow\n!color color_grade1 1b7837 variable\n!color color_grade2 5aae61\n!color color_grade3 a6dba0\n!color color_grade4 d9f0d3\n!color color_grade5 f7f7f7\n!color color_grade6 e7d4e8\n!color color_grade7 c2a5cf\n!color color_grade8 9970ab\n!color color_grade9 762a83 conserved"

    # design the seq3d to be printed out to the pipe file
    seq3d_grades_isd = design_string_for_pipe(seq3d_grades_isd)
    seq3d_grades = design_string_for_pipe(seq3d_grades)

    # creating the frequencies array which corresponds the number of residues in each grade
    [consurf_grade_freqs_isd, consurf_grade_freqs] = freq_array(ref_isd_residue_color, ref_no_isd_residue_color)

    # Taking Care of Strings
    if max_num_homol == "all":

        max_num_homol = "\"all\""

    # write to the pipe file
    try:

        PIPE = open(pipe_file, 'w')

    except:

        LOGGER.error("create_part_of_pipe_new : cannot open the file " + pipe_file + " for writing.", 'PANIC')

    PIPE.write("""! consurf_psi_blast_e_value = %s;
! consurf_psi_blast_database = "%s";
! consurf_psi_blast_iterations = %s;
! consurf_max_seqs = %s;
! consurf_apd = %.2f;
! consurf_alignment = "%s";
! consurf_method = "%s";
! consurf_substitution_model =  "%s";
!
! consurf_seqres_length = %s;
! consurf_atom_seq_length = %s;
! consurf_unique_seqs = %s;
! consurf_grade_freqs_isd = %s;
! consurf_grade_freqs = %s;
!
! seq3d_grades_isd =
%s
!
! seq3d_grades =
%s
!
!
!! ====== CONTROL PANEL OPTIONS SECTION ======
!js.init
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! pipe_title_enlarged = false;
! pipe_background_color = "white";
!
!! Specify the custom consurf control panel
!!
! pipe_cp1 = "consurf/consurf.htm";
!
!! If you want the frontispiece to be reset every time you enter this
!! page, use false. If this is a one-page presentation (no contents)
!! and you want to be able to return from QuickViews without resetting
!! the view, use true.
!!
! frontispiece_conditional_on_return = true;
!
!! Open the command input slot/message box to 30%% of window height.
!!
! pipe_show_commands = true;
! pipe_show_commands_pct = 30;
!
!! Don't show the PiPE presentation controls in the lower left frame.
!!
! pipe_hide_controls = true;
!
!! Hide development viewing mode links at the bottom of the control panel.
!!
! pipe_tech_info = false;
!
!! pipe_start_spinning = true; // default is PE's Preference setting.
!! top.nonStopSpin = true; // default: spinning stops after 3 min.
!!
!! ====== COLORS SECTION ======
!!
!color color_carbon C8C8C8
!color color_sulfur FFC832
!
!! Ten ConSurf color grades follow:
!!
%s
!
!
!! ====== SCRIPTS SECTION ======
!!----------------------------------------
!!
!spt #name=select_and_chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!!----------------------------------------
!!
!spt #name=view01
! @spt consurf_view_isd
!
!!----------------------------------------
!!
!spt #name=hide_all
! restrict none
! ssbonds off
! hbonds off
! dots off
! list * delete
!
!!----------------------------------------
!! common_spt uses CPK carbon gray (or phosphorus yellow) for backbones.
!!
!spt #name=common_spt
! @spt hide_all
! select all
! color [xC8C8C8] # rasmol/chime carbon gray
! select nucleic
! color [xFFA500] # phosphorus orange
! select hetero
! color cpk
! select not hetero
! backbone 0.4
! javascript top.water=0
!
! ssbonds 0.3
! set ssbonds backbone
! color ssbonds @color_sulfur
!
! select hetero and not water
! spacefill 0.45
! wireframe 0.15
! dots 50
!
! select protein
! center selected
!
!!----------------------------------------
!!
!spt #name=consurf_view_isd
! @spt common_spt
! @for $=0, 9
! @spt select_isd_grade$
! @spt select_and_chain
! color @color_grade$
! spacefill
! @endfor
! zoom 115
!
!!----------------------------------------
""" %(E_score, db, iterations, max_num_homol, round(float(Average_pairwise_distance), 2), MSAprogram, algorithm, matrix, length_of_seqres, length_of_atom, unique_seqs, consurf_grade_freqs_isd, consurf_grade_freqs, seq3d_grades_isd, seq3d_grades, scale_block))

    lineToPrint = ""
    i = 9
    while i > 0:

        PIPE.write("!!\n!spt #name=select_isd_grade%d\n!\n" %i)
        lineToPrint = print_selected(ref_isd_residue_color[i], "yes")
        if re.search(r'select', lineToPrint):

            PIPE.write(lineToPrint + "\n")

        PIPE.write("!\n!\n!!----------------------------------------\n")
        i -= 1

    PIPE.write("!!\n!spt #name=select_isd_grade0\n")
    lineToPrint = print_selected(ref_isd_residue_color[10], "yes")
    if re.search(r'select', lineToPrint):

        PIPE.write(lineToPrint + "\n")

    PIPE.write("!\n!\n!!----------------------------------------\n")

    i = 9
    while i > 0:

        PIPE.write("!!\n!spt #name=select_grade%d\n!\n" %i)
        lineToPrint = print_selected(ref_no_isd_residue_color[i], "yes")
        if re.search(r'select', lineToPrint):

            PIPE.write(lineToPrint + "\n")

        PIPE.write("!\n!\n!!----------------------------------------\n")
        i -= 1

    PIPE.write("!!\n!spt #name=select_grade0\n! select none\n!!\n")
    PIPE.write("!! ====== END OF CONSURF PiPE BLOCK ======\n")
    PIPE.close()


def design_string_for_pipe(string_to_format):
    # take a string aaaaaaa and returns it in this format: ! "aaa" +\n! "aa";\n
    part = string_to_format
    newPart = ""
    while len(part) > 73:
        newPart += "! \"" + part[:73] + "\" +\n"
        part = part[73:]
    newPart += "! \"" + part + "\" ;"
    return newPart

def freq_array(isd_residue_color, no_isd_residue_color):
    # design the frequencies array
    consurf_grade_freqs_isd = "Array(" + str(len(isd_residue_color[10]))
    i = 1
    while i < 10:
        consurf_grade_freqs_isd += "," + str(len(isd_residue_color[i]))
        i += 1
    consurf_grade_freqs_isd += ")"
    consurf_grade_freqs = "Array(0"
    i = 1
    while i < 10:
        consurf_grade_freqs += "," + str(len(no_isd_residue_color[i]))
        i += 1
    consurf_grade_freqs += ")"
    return(consurf_grade_freqs_isd, consurf_grade_freqs)

def print_selected(arr_ref, print_for_pipe):
    total_text = ""
    string = ""
    if print_for_pipe == "yes":
        string = "! select "
    else:
        string = "select "
    total_length = len(string)
    if len(arr_ref) > 0:
        for aa in arr_ref:
            aa = aa.replace(":", "")
            total_length += len(aa)
            if total_length > 80:
                if re.search(r', $', string):
                    string = string[:-2]
                total_text += string + "\n"
                if print_for_pipe == "yes":
                    string = "! select selected or %s, " %aa
                else:
                    string = "select selected or %s, " %aa
                total_length = len(string)
            else:
                string += aa + ", "
                total_length += 2
    else:
        total_text += string + "none"
    if re.search(r', $', string):
        string = string[:-2]
        total_text += string
    return total_text

def extract_data_from_pdb(input_pdb_file):

    header = ""
    title = ""
    compnd = ""
    try:
        PDB = open(input_pdb_file, 'r')
    except:
        LOGGER.error("extract_data_from_pdb : Could not open the file " + input_pdb_file + " for reading.")
    line = PDB.readline()
    while line != "":
        match1 = re.match(r'^HEADER', line)
        if match1:
            header = line.rstrip()
        else:
            match2 =re.match(r'^TITLE\s+\d*\s(.*)', line)
            if match2:
                title += match2.group(1) + " "
            else:
                match3 = re.match(r'^COMPND\s+\d*\s(.*)', line)
                if match3:
                    compnd += match3.group(1) + " "
                elif re.match(r'^SOURCE', line) or re.match(r'^KEYWDS', line) or re.match(r'^AUTHOR', line) or re.match(r'^SEQRES', line) or re.match(r'^ATOM', line):
                    break # no nead to go over all the pdb
        line = PDB.readline()
    PDB.close()
    if title == "":
        return header, compnd
    else:
        return header, title

def create_consurf_pipe_new(results_dir, IN_pdb_id_capital, chain, ref_header_title, final_pipe_file, identical_chains, partOfPipe, current_dir, run_number, msa_filename, query_name_in_msa = "", tree_filename = "", submission_time = "", completion_time = "", run_date = ""):


    # Create the pipe file for FGiJ

    if chain == 'NONE':

        chain = ""
        identical_chains = ""

    # read info from the pdb file
    [header_line, title_line] = ref_header_title

    if title_line == "":

        title_line = "! \"No title or compound description was found in the PDB file\";"

    else:

        title_line = design_string_with_spaces_for_pipe(title_line)

    # design the identical chains line
    identical_chains_line = "! consurf_identical_chains = \"%s\";" %identical_chains

    current_dir += "_sourcedir"
    # in case there is a source dir - we determine the var query_name_in_msa
    if os.path.exists(current_dir):

        try:

            SOURCEDIR = open(current_dir, 'r')

        except:

            LOGGER.error("create_consurf_pipe : cannot open " + current_dir + " for reading.")

        match = re.match(r'(\d[\d\w]{3})\/(\w)', SOURCEDIR.readline())
        if match:

            query_name_in_msa = SOURCEDIR.group(1) + SOURCEDIR.group(2)
            SOURCEDIR.close()

    if query_name_in_msa == "":

        query_name_in_msa = IN_pdb_id_capital + chain.upper()

    # write to the pipe file
    try:

        PIPE_PART = open(partOfPipe, 'r')

    except:

        LOGGER.error("create_consurf_pipe : cannot open " + partOfPipe + " for reading.")

    try:

        PIPE = open(final_pipe_file, 'w')

    except:

        LOGGER.error("create_consurf_pipe : cannot open " + final_pipe_file + " for writing.")

    if header_line != "":

        PIPE.write(header_line + "\n")

    else:

        PIPE.write("HEADER                                 [THIS LINE ADDED FOR JMOL COMPATIBILITY]\n")

    PIPE.write("""!! ====== IDENTIFICATION SECTION ======
!js.init
! consurf_server = "consurf";
! consurf_version = "3.0";
! consurf_run_number = \"%s\";
! consurf_run_date = \"%s\";
! consurf_run_submission_time = \"%s\";
! consurf_run_completion_time = \"%s\";
!
! consurf_pdb_id = \"%s\";
! consurf_chain = \"%s\";
%s
! consurf_msa_filename = \"%s\";
! consurf_msa_query_seq_name = \"%s\";
! consurf_tree_filename = \"%s\";
!
""" %(run_number, run_date, submission_time, completion_time, IN_pdb_id_capital, chain, identical_chains_line, msa_filename, query_name_in_msa, tree_filename))

    titleFlag = 0
    line = PIPE_PART.readline()
    while line != "":

        if re.match(r'^~~~+', line):

            if titleFlag == 0:

                PIPE.write("! pipe_title = \"<i>ConSurf View:</i> %s chain %s.\"\n!! pipe_subtitle is from TITLE else COMPND\n!!\n! pipe_subtitle =\n%s\n" %(IN_pdb_id_capital, chain, title_line))
                titleFlag = 1

            elif chain != "":

                PIPE.write("! select selected and :%s\n" %chain)

            else:

                PIPE.write("! select selected and protein\n")

        else:

            PIPE.write(line)

        line = PIPE_PART.readline()

    PIPE_PART.close()
    PIPE.close()

def design_string_with_spaces_for_pipe(part_input):
    if part_input.strip() == "":
        return ""
    
    words = part_input.split()
    newPart = "! \"" +words[0]
    part = ""
    for word in words[1:]:

        # if adding another word to the string will yeild a too long string - we cut it.
        if len(word) + 1 + len(newPart) > 76:

            part += newPart + " \" +\n"
            newPart = "! \"" + word

        else:

            newPart += " " + word

    part += newPart + "\" ;"

    return part

def add_pdb_data_to_pipe(pdb_file, pipe_file):

    # create the file to be shown using FGiJ. read the pdb file and concat header pipe to it.
    try:
        PIPE = open(pipe_file, 'a')
    except:
        LOGGER.error("add_pdb_data_to_pipe: cannot open " + pipe_file + " for writing.")
    try:
        PDB_FILE = open(pdb_file, 'r')
    except:
        LOGGER.error("add_pdb_data_to_pipe: cannot open the " + pdb_file + " for reading.")
    line = PDB_FILE.readline()
    while line != "":
        if not re.match(r'^HEADER', line):
            PIPE.write(line)
        line = PDB_FILE.readline()
    PIPE.close()
    PDB_FILE.close()

def create_rasmol(job_name, chain, ref_colors_array, ref_colors_array_isd, vars):
    
    RasMol_file = job_name + "_jmol_consurf_colors.spt"
    RasMol_file_isd = job_name + "_jmol_consurf_colors_isd.spt"
    RasMol_file_CBS = job_name + "_jmol_consurf_colors_CBS.spt"
    RasMol_file_CBS_isd = job_name + "_jmol_consurf_colors_CBS_isd.spt"
	
    if chain == "NONE":
        
        chain = " "
        
    print_rasmol(job_name, RasMol_file, False, ref_colors_array, chain, False, vars)
    print_rasmol(job_name, RasMol_file_CBS, False, ref_colors_array, chain, True, vars)

    if len(ref_colors_array_isd[10]) > 0: # there is isd
        
        print_rasmol(job_name, RasMol_file_isd, True, ref_colors_array_isd, chain, False, vars)
        print_rasmol(job_name, RasMol_file_CBS_isd, True, ref_colors_array_isd, chain, True, vars)

def print_rasmol(job_name, out_file, isd, ref_colors_array, chain, cbs, vars):

    # print out new format of rasmol
    consurf_rasmol_colors = ["", "[16,200,209]", "[140,255,255]", "[215,255,255]", "[234,255,255]", "[255,255,255]", "[252,237,244]", "[250,201,222]", "[240,125,171]", "[160,37,96]", "[255,255,150]"]
    consurf_rasmol_colors_CBS = ["", "[27,120,55]", "[90,174,97]", "[166,219,160]", "[217,240,211]", "[247,247,247]", "[231,212,232]", "[194,165,207]", "[153,112,171]", "[118,42,131]", "[255,255,150]"]

    try:
        OUT = open(out_file, 'w')
    except:
        LOGGER.error("print_rasmol : Could not open the file " + out_file + " for writing.")

    OUT.write("ConSurfDB.tau.ac.il   %s   %s\nAPD N.NN\n" %(vars['date'], job_name))
    OUT.write("select all\ncolor [200,200,200]\n\n")
    
    i = len(ref_colors_array) - 1
    while i > 0:
        if i == 10 and not isd:
            i -= 1
            continue
        if len(ref_colors_array[i]) > 0:
            OUT.write(print_selected(ref_colors_array[i], "no"))
            OUT.write("\nselect selected and :%s\n" %chain)
            if cbs:
                OUT.write("color %s\nspacefill\n" %consurf_rasmol_colors_CBS[i])
            else:
                OUT.write("color %s\nspacefill\n" %consurf_rasmol_colors[i])
            OUT.write("define CON%d selected\n\n" %i)
        i -= 1
    OUT.close()

def replace_TmpFactor_Consurf_Scores_CIF(atom_grades, query_chain, pdb_file, prefix, form, vars):

    # Creates The ATOM section with ConSurf grades instead of the TempFactor column, creates PDB file with ConSurf grades

    pdb_with_grades = prefix + "_ATOMS_section_With_ConSurf.cif"
    pdb_with_grades_isd = prefix + "_ATOMS_section_With_ConSurf_isd.cif"
    pdb_with_scores = prefix + "_With_Conservation_Scores.cif"
	
    [auth_seq_id_column, auth_comp_id_column, auth_asym_id_column, B_iso_or_equiv] = vars['pdb_object'].get_columns()

    try:
        PDB = open(pdb_file, 'r')
    except:
        LOGGER.error("could not open file '" + pdb_file + "' for reading.\n")

    try:
        GRADES = open(pdb_with_grades, 'w')
    except:
        LOGGER.error("could not open the file '" + pdb_with_grades + "' for writing.\n")

    try:
        SCORES = open(pdb_with_scores, 'w')
    except:
        LOGGER.error("could not open the file '" + pdb_with_scores + "' for writing.\n")
		
    if vars['insufficient_data']:
        try:
            GRADES_ISD = open(pdb_with_grades_isd, 'w')
        except:
            LOGGER.error("could not open the file '" + pdb_with_grades_isd + "' for writing.\n")

    line = PDB.readline()
    while line != "":
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            words = line.split()
            chain = words[auth_asym_id_column]
            residue_number = words[auth_seq_id_column]

            grade = "0"
            score = "0"
            grade_isd = "0"

            if residue_number in atom_grades and chain == query_chain:
                # getting the grade
                [grade, isd, score] = atom_grades[residue_number]

                if vars['insufficient_data']:
                    if isd == 1:
                        grade_isd = "10"
                    else:
                        grade_isd = grade
					
            match = re.match(r'^((\S+\s+){' + str(B_iso_or_equiv) + r'})(\S+\s+)(.+)', line)
            length_temp_fact = len(match.group(3))
            line_start = match.group(1)
            line_end = match.group(4)
            while len(grade) < length_temp_fact:
                grade = grade + " "
            while len(score) < length_temp_fact:
                score = score + " "
            while len(grade_isd) < length_temp_fact:
                grade_isd = grade_isd + " "

            GRADES.write(line_start + grade + line_end + "\n")
            SCORES.write(line_start + score + line_end + "\n")
            if vars['insufficient_data']:
                GRADES_ISD.write(line_start + grade_isd + line_end + "\n")
        else:
            GRADES.write(line)
            SCORES.write(line)
            if vars['insufficient_data']:
                GRADES_ISD.write(line)
        line = PDB.readline()

    GRADES.close()
    SCORES.close()	
    vars['zip_list'].append(pdb_with_grades)
    vars['zip_list'].append(pdb_with_scores)
	
    if vars['insufficient_data']:
        GRADES_ISD.close()
        vars['zip_list'].append(pdb_with_grades_isd)
        create_chimera(pdb_with_grades_isd, prefix + "_", form, vars)
        create_pymol(pdb_with_grades_isd, prefix + "_", form, vars)
    else:
        create_chimera(pdb_with_grades, prefix + "_", form, vars)
        create_pymol(pdb_with_grades, prefix + "_", form, vars)

def create_chimera(input, prefix, form, vars):
    run_chimera(input, prefix + "consurf_chimerax_session.cxs", vars['chimera_color_script'], form, vars)
    run_chimera(input, prefix + "consurf_CBS_chimerax_session.cxs", vars['chimera_color_script_CBS'], form, vars)

def run_chimera(input, output, script, form, vars):

    cmd = "chimerax --nogui --script '%s %s %s' --exit\n" %(script, input, output)
    LOGGER.info("create_chimera : %s\n" %cmd)
    submit_job_to_Q("CHIMERA", cmd)
    vars['zip_list'].append(output)

def create_pymol(input, prefix, form, vars):
    cmd = "pymol -qc " + input + " -d \"run " + vars['pymol_color_script_isd'] + "\"\n"
    cmd += "pymol -qc " + input + " -d \"run " + vars['pymol_color_script_CBS_isd'] + "\"\n"

    LOGGER.info("create_pymol : %s\n" %cmd)
    submit_job_to_Q("PYMOL", cmd)

    pymol_session = "consurf_pymol_session.pse"
    pymol_session_CBS = "consurf_CBS_pymol_session.pse"

    if os.path.exists(pymol_session) and os.path.getsize(pymol_session) != 0:

        os.chmod(pymol_session, 0o664)
        os.rename(pymol_session, prefix + pymol_session)
        vars['zip_list'].append(prefix + pymol_session)

    if os.path.exists(pymol_session_CBS) and os.path.getsize(pymol_session_CBS) != 0:

        os.chmod(pymol_session_CBS, 0o664)
        os.rename(pymol_session_CBS, prefix + pymol_session_CBS)
        vars['zip_list'].append(prefix + pymol_session_CBS)

def replace_TmpFactor_Consurf_Scores(atom_grades, chain, pdb_file, prefix, form, vars):

    # This Will create a File containing the ATOMS records with the ConSurf grades instead of the TempFactor column

    if vars['cif_or_pdb'] == "pdb":

        LOGGER.info("replace_TmpFactor_Consurf_Scores_PDB(atom_grades, %s, %s, %s);\n" %(chain, pdb_file, prefix))
        replace_TmpFactor_Consurf_Scores_PDB(atom_grades, chain, pdb_file, prefix, form, vars)
		
    else:																																												 
        LOGGER.info("replace_TmpFactor_Consurf_Scores_CIF(atom_grades, %s, %s, %s);\n" %(chain, pdb_file, prefix))
        replace_TmpFactor_Consurf_Scores_CIF(atom_grades, chain, pdb_file, prefix, form, vars)
    
def replace_TmpFactor_Consurf_Scores_PDB(atom_grades, query_chain, pdb_file, prefix, form, vars):
    # Creates The ATOM section with ConSurf grades instead of the TempFactor column, creates PDB file with ConSurf grades
    pdb_with_grades = prefix + "_ATOMS_section_With_ConSurf.pdb"
    pdb_with_grades_isd = prefix + "_ATOMS_section_With_ConSurf_isd.pdb"
    pdb_with_scores = prefix + "_With_Conservation_Scores.pdb"

    try:
        PDB = open(pdb_file, 'r')
    except:
        LOGGER.error("replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_file + "' for reading.\n")

    try:
        GRADES = open(pdb_with_grades, 'w')
    except:
        LOGGER.error("replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_grades + "' for writing.\n")

    try:
        SCORES = open(pdb_with_scores, 'w')
    except:
        LOGGER.error("replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_scores + "' for writing.\n")

    if vars['insufficient_data']:
        try:
            GRADES_ISD = open(pdb_with_grades_isd, 'w')
        except:
            LOGGER.error("replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_grades_isd + "' for writing.\n")

    line = PDB.readline()
    while line != "":
        if line.strip() == "":
            # line is empty
            line = PDB.readline()
            continue
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            chain = line[21:22]
            if chain == " ":
                chain = "NONE"
            residue = (line[22:27]).strip()
            if residue in atom_grades and chain == query_chain:
                [grade, isd, score] = atom_grades[residue]
                while len(score) < 6:
                    score = " " + score
                # the TF is updated with the grades and scores
                GRADES.write(line[:60] + "     " + grade + "      \n")
                SCORES.write(line[:60] + score + line[66:])
                if vars['insufficient_data']:
                    # the TF is updated with the number from gradesPE showing isd
                    if isd == 1:
                        GRADES_ISD.write(line[:60] + "    10      \n")
                    else:
                        GRADES_ISD.write(line[:60] + "    " + grade + "      \n")
            else:
                GRADES.write(line[:60] + "            \n")
                SCORES.write(line[:60] + "            \n")
                if vars['insufficient_data']:
                    GRADES_ISD.write(line[:60] + "            \n")
        else:
            GRADES.write(line)
            SCORES.write(line)
            if vars['insufficient_data']:
                GRADES_ISD.write(line)
        line = PDB.readline()

    GRADES.close()
    SCORES.close()	
    vars['zip_list'].append(pdb_with_grades)
    vars['zip_list'].append(pdb_with_scores)
	
    if vars['insufficient_data']:
        GRADES_ISD.close()
        vars['zip_list'].append(pdb_with_grades_isd)
        create_chimera(pdb_with_grades_isd, prefix + "_", form, vars)
        create_pymol(pdb_with_grades_isd, prefix + "_", form, vars)
    else:
        create_chimera(pdb_with_grades, prefix + "_", form, vars)
        create_pymol(pdb_with_grades, prefix + "_", form, vars)

def create_pdf(vars):
    create_pdf_regular_or_cbs(True, vars['Colored_Seq_PDF'], vars)
    create_pdf_regular_or_cbs(False, vars['Colored_Seq_CBS_PDF'], vars)

def create_pdf_regular_or_cbs(cbs, name, vars):
    #prediction_method = prediction_method.replace('-', ' ')
    pdf = PDF()
    pdf.add_page()
    #pdf.set_font("Times", "B", 20)
    pdf.add_font('DejaVu', '', GENERAL_CONSTANTS.FONTS, uni=True)
    pdf.set_font('DejaVu', '', 20)
    pdf.Cell(0, 0, "ConSurf Results. Date:" + vars['date'], 0, 0, 'C')
    pdf.set_font("Times", "B", 20)
    pdf.set_y(pdf.get_y() + 10)        
    pdf.cbs = cbs
    Rows_Pos = 0
	
    maxPosPerPage = 600
    for elem in vars['gradesPE_Output']:
        Pos = elem['POS'] - 1
        if vars['B/E']:
            prediction_method = "neural network algorithm"
            B_E = elem['B/E']
            F_S = elem['F/S'].strip()
        else:
            prediction_method = "no prediction"
            B_E = ""
            F_S = ""
			
        if Pos % maxPosPerPage == 0 and Pos != 0:
            pdf.add_page()
        if Pos % 50 == 0:
            pdf.ln()
            pdf.ln()
            pdf.ln()
            pdf.ln()
            Rows_Pos = pdf.get_y()
        elif Pos % 10 == 0:
            pdf.Print_ForegroundColor("", 'B', 10, 0, 4)
        if Pos % 10 == 0:
            pdf.Print_4_Lines_Element(Rows_Pos, elem['COLOR'], elem['SEQ'], Pos, B_E, 10, 0, F_S)
        else:
            pdf.Print_4_Lines_Element(Rows_Pos, elem['COLOR'], elem['SEQ'], "", B_E, 10, 0, F_S)
            
    pdf.ln()
    pdf.ln()
    pdf.ln()
    pdf.ln()
    pdf.Print_NEW_Legend(vars['B/E'], vars['B/E'], vars['insufficient_data'],  prediction_method)
    
    pdf.output(name)

class PDF(fpdf.FPDF):
    
    def __init__(self):
        super().__init__()
        self.lasth = 0 
        self.cbs = False
        
    def Cell(self, w, h = 0, txt = '', border = 0, ln = 0, align = '', fill = False):
        self.cell(w, h, txt, border, ln, align, fill)
        self.lasth = h

    def Print_4_Lines_Element(self, Rows_Pos, Score, AA, Pos, Solv_Acc, font_size, Insufficient_Data, Funct_Struct = ""):
        if Pos != "":
            x = self.get_x()
            if Pos < 9:
                self.set_xy(self.get_x() + 0.5, self.get_y())
            elif Pos < 99:
                self.set_xy(self.get_x() + 1, self.get_y())
            elif Pos < 9999:
                self.set_xy(self.get_x() + 1.5, self.get_y())
            elif Pos < 99999:
                self.set_xy(self.get_x() + 2, self.get_y())
            self.Print_BackgroundColor(str(Pos + 1), "", font_size, 5)
            self.set_xy(x, self.get_y())
            
        self.set_xy(self.get_x(), self.get_y() + self.lasth)
        self.Print_BackgroundColor(AA, 'B', font_size, Score, 3, Insufficient_Data)
        Col_Pos = self.get_x() # position on the line after printing
        self.set_xy(self.get_x() - 2.5, self.get_y() + self.lasth + 0.2)
        if Solv_Acc == "e":
            self.Print_ForegroundColor(Solv_Acc, 'B', font_size, 1)
        elif Solv_Acc == "b":
            self.Print_ForegroundColor(Solv_Acc, 'B', font_size, 2)
        self.set_xy(self.get_x() - 2, self.get_y() + self.lasth - 1.2)
        if Funct_Struct == "f":
            self.Print_ForegroundColor(Funct_Struct, 'B', font_size, 3)
        elif Funct_Struct == "s":
            self.Print_ForegroundColor(Funct_Struct, 'B', font_size, 4)
        self.set_xy(Col_Pos, Rows_Pos)

    def Print_BackgroundColor(self, txt, print_style, font_size, Color_Num, spacer = 2, isd = False):
        cbs = self.cbs
        if Color_Num == 0:
            self.set_text_color(0, 0, 0)
            self.set_fill_color(255, 255, 150)
        elif Color_Num == 1:
            if cbs:
                self.set_fill_color(10, 125, 130)
            else:
                self.set_fill_color(15, 90, 35)
            self.set_text_color(255, 255, 255)
        elif Color_Num == 2:
            if cbs:
                self.set_fill_color(75, 175, 190)
            else:
                self.set_fill_color(90, 175, 95)
        elif Color_Num == 3:
            if cbs:
                self.set_fill_color(165, 220, 230)
            else:
                self.set_fill_color(165, 220, 160)
        elif Color_Num == 4:
            if cbs:
                self.set_fill_color(215, 240, 240)
            else:
                self.set_fill_color(215, 240, 210)
        elif Color_Num == 5:
            self.set_fill_color(255, 255, 255)
        elif Color_Num == 6:
            if cbs:
                self.set_fill_color(250, 235, 245)
            else:
                self.set_fill_color(230, 210, 230)
        elif Color_Num == 7:
            if cbs:
                self.set_fill_color(250, 200, 220)
            else:
                self.set_fill_color(195, 165, 205)
        elif Color_Num == 8:
            if cbs:
                self.set_fill_color(240, 125, 170)
            else:
                self.set_fill_color(155, 110, 170)
                
        elif Color_Num == 9:
            if cbs:
                self.set_fill_color(160, 40, 95)
            else:
                self.set_fill_color(120, 40, 130)
            self.set_text_color(255, 255, 255)
        if isd:
            self.set_fill_color(255, 255, 150)

        self.set_font("Courier", print_style, font_size)
        width = len(txt) - 1 + spacer
        high = font_size / 2
        self.Cell(width, high, txt, 0, 0, "C", True)
        self.set_fill_color(255, 255, 255) # return to default background color (white)
        self.set_text_color(0, 0, 0) # return to default text color (black)
        
    def Print_NEW_Legend(self, IS_THERE_FUNCT_RES, IS_THERE_STRUCT_RES, IS_THERE_INSUFFICIENT_DATA, B_E_METHOD):
        self.set_font("", 'B', 12)
        font_size = 12
        self.ln()
        self.Cell(40, 10, "The conservation scale:", 0, 1)
        self.set_xy(18, self.get_y())
        #self.Print_BackgroundColor('?', "", 12, 0, 4, 1)
        self.Print_BackgroundColor('1', "", 12, 1, 4)
        self.Print_BackgroundColor('2', "", 12, 2, 4)
        self.Print_BackgroundColor('3', "", 12, 3, 4)
        #Average_X = self.get_x()
        self.Print_BackgroundColor('4', "", 12, 4, 4)
        self.Print_BackgroundColor('5', "", 12, 5, 4)
        self.Print_BackgroundColor('6', "", 12, 6, 4)
        self.Print_BackgroundColor('7', "", 12, 7, 4)
        #Conserved_X = self.get_x()
        self.Print_BackgroundColor('8', '', 12, 8, 4)
        self.Print_BackgroundColor('9', '', 12, 9, 4)
        self.ln()
        self.set_font("Times", 'B', 9.5)
        self.Cell(10, 6, "Variable", 0, 0, 'R', False)
        self.set_xy(28, self.get_y())
        self.Cell(15, 6, "Average", 0, 0, 'R', False)
        self.set_xy(53, self.get_y())
        self.Cell(15, 6, "Conserved", 0, 0, 'R', False)
        self.ln()
        self.ln()
    
        if B_E_METHOD != "no prediction":
            offset_1 = 0 
            offset_2 = 0
            if B_E_METHOD == "neural network algorithm":
                offset_1 = 63.5
                offset_2 = 62
            elif B_E_METHOD == "NACSES algorithm":
                offset_1 = 57.5
                offset_2 = 56
            self.Print_ForegroundColor('e', 'B', font_size, 1)
            self.set_xy(offset_1, self.get_y())
            self.Print_ForegroundColor(" - An exposed residue according to the %s." %B_E_METHOD, 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.set_x(self.left_margin)
            self.Print_ForegroundColor('b', 'B', font_size, 2)
            self.set_xy(offset_2, self.get_y())
            self.Print_ForegroundColor(" - A buried residue according to the %s." %B_E_METHOD, 'B', font_size, 9)
            self.ln()
             
        #self.set_xy(45, self.get_y() + self.lasth + 5)
        #self.y += self.lasth + 5
        #self.x = self.left_margin
        if IS_THERE_FUNCT_RES:
            self.Print_ForegroundColor('f', 'B', font_size, 3)
            self.set_xy(64.5, self.get_y())
            self.Print_ForegroundColor(" - A predicted functional residue (highly conserved and exposed).", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
        if IS_THERE_STRUCT_RES:
            self.Print_ForegroundColor('s', 'B', font_size, 4)
            self.set_xy(64, self.get_y())
            self.Print_ForegroundColor(" - A predicted structural residue (highly conserved and buried).", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
        if IS_THERE_INSUFFICIENT_DATA:
            self.Print_BackgroundColor('x', 'B', font_size, 0, 2, 1)
            self.set_xy(58, self.get_y())
            self.Print_ForegroundColor(" - Insufficient data - the calculation for this site was", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 4)
            #self.x = self.left_margin
            self.set_xy(48, self.get_y())
            self.Print_ForegroundColor("     performed on less than 10% of the sequences.",'B', font_size, 9)
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
            
    def Print_ForegroundColor(self, txt, print_style, font_size, Color, spacer = 2):
        if Color == 1: # orange
            self.set_text_color(255, 153, 0)
        elif Color == 2: # green
            self.set_text_color(0, 204, 0)
        elif Color == 3: # red
            self.set_text_color(255, 0, 0)
        elif Color == 4: # blue
            self.set_text_color(0, 0, 153)
        else: # black
            self.set_text_color(0, 0, 0)
        self.set_font("Courier", print_style, font_size)
        width = len(txt) - 1 + spacer
        high = font_size / 2
        self.Cell(width, high, txt, 0, 0, 'C', True)
        self.set_fill_color(255, 255, 255) # return to default background (white)
        self.set_text_color(0, 0, 0) # return to default text color (black)

def conseq_create_output(form, vars):
    create_gradesPE(vars['gradesPE'], form=form, vars=vars)
    # create_pdf(vars)

def zip_all_outputs(vars):
    zipObj = ZipFile(vars['All_Outputs_Zip'], 'w')
    for file in vars['zip_list']:
        if os.path.exists(file):
            zipObj.write(file)	
    zipObj.close()

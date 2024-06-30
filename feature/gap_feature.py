import Bio.SeqIO as Seq
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
import sys
from tqdm import tnrange, trange, tqdm, tqdm_notebook
import time

def _is_kernel():
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None


def my_tqdm():
    return tqdm_notebook if _is_kernel() else tqdm


def my_trange():
    return tnrange if _is_kernel() else trange


######################################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:10:49 2016

@author: jessime
"""


class Reader():
    """Fixes any compatibility issues a fasta file might have with this code.

    Parameters
    ----------
    infasta : str (default=None)
        Name of input fasta file to be manipulated
    outfasta : str (default=None)
        location to store extracted data from infasta
    names : iter (default=None)
        Common style names to use in header lines

    Attributes
    ----------
    data : list
        Raw lines of the infasta file
        Note: This is different than the data attribute in other classes

    Examples
    --------
    Putting the sequence on one line instead of breaking it every 80 chars.
    Making sure the whole sequence is capitalized.
    Restructuring the name line to work with GENCODE's naming.
    """

    def __init__(self, infasta=None, outfasta=None, names=None):
        self.infasta = infasta
        self.outfasta = outfasta
        self.names = names

        self.data = None

    def _read_data(self):
        """Sets data to stripped lines from the fasta file
        """
        with open(self.infasta) as infasta:
            self.data = [l.strip() for l in infasta.readlines()]

    def _upper_seq_per_line(self):
        """Sets data to upper case, single line sequences for each header
        """
        new_data = []
        seq = ''
        for i, line in enumerate(self.data):
            if line[0] == '>':
                if seq:
                    new_data.append(seq.upper())
                    seq = ''
                else:
                    assert i == 0, 'There may be a header without a sequence at line {}.'.format(i)
                new_data.append(line)
            else:
                seq += line
        new_data.append(seq.upper())
        self.data = new_data

    def get_lines(self):
        self._read_data()
        self._upper_seq_per_line()
        return self.data

    def get_seqs(self):
        clean_data = self.get_lines()
        seqs = clean_data[1::2]
        return seqs

    def get_headers(self):
        clean_data = self.get_lines()
        headers = clean_data[::2]
        return headers

    def get_data(self, tuples_only=False):
        clean_data = self.get_lines()
        headers = clean_data[::2]
        seqs = clean_data[1::2]
        tuples = zip(headers, seqs)
        if tuples_only:
            return tuples
        else:
            return tuples, headers, seqs

    def supply_basic_header(self):
        """Convert headerlines to GENCODE format with only common name and length"""
        new_fasta = []

        if self.names is None:
            self.names = iter(self.get_headers())
        for i, line in enumerate(self.data):
            if line[0] == '>':
                name = next(self.names).strip('>')
                length = len(self.data[i + 1])
                new_fasta.append('>||||{}||{}|'.format(name, length))
            else:
                new_fasta.append(line)
        return new_fasta

    def save(self):
        """Write self.data to a new fasta file"""
        with open(self.outfasta, 'w') as outfasta:
            for line in self.data:
                outfasta.write(line + '\n')


##############################################################################
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description
-----------
Generates a kmer count matrix of m rows by n columns,
where m is the number of transcripts in a fasta file and n is 4^kmer.

Examples
--------
The default settings produce a binary, normalized numpy file:
    $ python kmer_counts.py /path/to/rnas.fa -o /path/to/out.npy

To get a human readable csv file, set the nonbinary flag:
    $ python kmer_counts.py /path/to/rnas.fa -o /path/to/out.csv -nb

If you want to add default labels, also set the label flag:
    $ python kmer_counts.py /path/to/rnas.fa -o /path/to/out.csv -nb -lb

You can change also change the size of the kmer you're using, and prevent normalization:
    $ python kmer_counts.py /path/to/rnas.fa -o /path/to/out.npy -k 4 -nc -ns

Notes
-----
For more sophisticated options, you cannot use the command-line, but need python instead.
To label the axes of the matrix, for example, you can call BasicCounter('/path/rnas.fa').to_csv(names)

Issues
------
Any issues can be reported to https://github.com/CalabreseLab #TODO

---
"""

class BasicCounter:
    """Generates overlapping kmer counts for a fasta file

    Parameters
    ----------
    infasta : str (default=None)
        Full path to fasta file to be counted
    outfile : str (default=None)
        Full path to the counts file to be saved
    k : int (default=6)
        Size of kmer to be counted
    binary : bool (default=True)
        Saves as numpy array if True, else saves as csv
    mean : bool, np.array, str (default=True)
        Set the mean to 0 for each kmer/column of the count matrix.
        If str, provide path to a previously calculated mean array.
    std : bool or str (default=True)
        Set the std. dev. to 1 for each kmer/column of the count matrix
        If str, provide path to a previously calculated std array.
    leave : bool (default=True)
        Set to False if get_counts is used within another tqdm loop
    silent : bool (default=False)
        Set to True to turn off tqdm progress bar

    Attributes
    ----------
    counts : None
        Stores the ndarray of kmer counts
    kmers : list
        str elements of all kmers of size k
    map : dict
        Mapping of kmers to column values
    """

    def __init__(self, infasta=None, k=6,
                 binary=True, mean=False, std=False,
                 leave=True, silent=False, label=False):
        self.infasta = infasta
        self.seqs = None
        if infasta is not None:
            self.seqs = Reader(infasta).get_seqs()
        # self.outfile = outfile
        self.k = k
        self.binary = binary
        self.mean = mean
        if isinstance(mean, str):
            self.mean = np.load(mean)
        self.std = std
        if isinstance(std, str):
            self.std = np.load(std)
        self.leave = leave
        self.silent = silent
        self.label = label

        self.counts = None
        self.kmers = [''.join(i) for i in product('AGTC', repeat=k)]
        self.map = {k: i for k, i in zip(self.kmers, range(4 ** k))}

        if len(self.seqs) == 1 and self.std is True:
            err = ('You cannot standardize a single sequence. '
                   'Please pass the path to an std. dev. array, '
                   'or use raw counts by setting std=False.')
            raise ValueError(err)

    def occurrences(self, row, seq):
        """Counts kmers on a per kilobase scale"""
        counts = defaultdict(int)
        length = len(seq)
        increment = 1000 / length
        for c in range(length - self.k + 1):
            kmer = seq[c:c + self.k]
            counts[kmer] += increment
        for kmer, n in counts.items():
            if kmer in self.map:
                row[self.map[kmer]] = n
        return row

    def _progress(self):
        """Determine which iterator to loop over for counting."""
        if self.silent:
            return self.seqs

        if not self.leave:
            tqdm_seqs = my_tqdm()(self.seqs, desc='Kmers', leave=False)
        else:
            tqdm_seqs = my_tqdm()(self.seqs)

        return tqdm_seqs

    def center(self):
        """mean center counts by column"""
        if self.mean is True:
            self.mean = np.mean(self.counts, axis=0)
        self.counts -= self.mean

    def standardize(self):
        """divide out the standard deviations from columns of the count matrix"""
        if self.std is True:
            self.std = np.std(self.counts, axis=0)
        self.counts /= self.std

    def get_counts(self):
        """Generates kmer counts for a fasta file"""
        self.counts = np.zeros([len(self.seqs), 4 ** self.k], dtype=np.float32)
        seqs = self._progress()
        for i, seq in enumerate(seqs):
            self.counts[i] = self.occurrences(self.counts[i], seq)
        if self.mean is not False:
            self.center()
        if self.std is not False:
            self.standardize()

        seqname = []
        for seq in Seq.parse(self.infasta, 'fasta'):
            seqid = seq.id
            seqname.append(seqid)
        if self.k == 3:
            columnanmes = ['KMAAA: Transcript k-mer AAA content','KMAAG: Transcript k-mer AAG content','KMAAT: Transcript k-mer AAT content','KMAAC: Transcript k-mer AAC content','KMAGA: Transcript k-mer AGA content','KMAGG: Transcript k-mer AGG content','KMAGT: Transcript k-mer AGT content','KMAGC: Transcript k-mer AGC content','KMATA: Transcript k-mer ATA content','KMATG: Transcript k-mer ATG content','KMATT: Transcript k-mer ATT content','KMATC: Transcript k-mer ATC content','KMACA: Transcript k-mer ACA content','KMACG: Transcript k-mer ACG content','KMACT: Transcript k-mer ACT content','KMACC: Transcript k-mer ACC content','KMGAA: Transcript k-mer GAA content','KMGAG: Transcript k-mer GAG content','KMGAT: Transcript k-mer GAT content','KMGAC: Transcript k-mer GAC content','KMGGA: Transcript k-mer GGA content','KMGGG: Transcript k-mer GGG content','KMGGT: Transcript k-mer GGT content','KMGGC: Transcript k-mer GGC content','KMGTA: Transcript k-mer GTA content','KMGTG: Transcript k-mer GTG content','KMGTT: Transcript k-mer GTT content','KMGTC: Transcript k-mer GTC content','KMGCA: Transcript k-mer GCA content','KMGCG: Transcript k-mer GCG content','KMGCT: Transcript k-mer GCT content','KMGCC: Transcript k-mer GCC content','KMTAA: Transcript k-mer TAA content','KMTAG: Transcript k-mer TAG content','KMTAT: Transcript k-mer TAT content','KMTAC: Transcript k-mer TAC content','KMTGA: Transcript k-mer TGA content','KMTGG: Transcript k-mer TGG content','KMTGT: Transcript k-mer TGT content','KMTGC: Transcript k-mer TGC content','KMTTA: Transcript k-mer TTA content','KMTTG: Transcript k-mer TTG content','KMTTT: Transcript k-mer TTT content','KMTTC: Transcript k-mer TTC content','KMTCA: Transcript k-mer TCA content','KMTCG: Transcript k-mer TCG content','KMTCT: Transcript k-mer TCT content','KMTCC: Transcript k-mer TCC content','KMCAA: Transcript k-mer CAA content','KMCAG: Transcript k-mer CAG content','KMCAT: Transcript k-mer CAT content','KMCAC: Transcript k-mer CAC content','KMCGA: Transcript k-mer CGA content','KMCGG: Transcript k-mer CGG content','KMCGT: Transcript k-mer CGT content','KMCGC: Transcript k-mer CGC content','KMCTA: Transcript k-mer CTA content','KMCTG: Transcript k-mer CTG content','KMCTT: Transcript k-mer CTT content','KMCTC: Transcript k-mer CTC content','KMCCA: Transcript k-mer CCA content','KMCCG: Transcript k-mer CCG content','KMCCT: Transcript k-mer CCT content','KMCCC: Transcript k-mer CCC content']
        else:
            columnanmes = self.kmers
        df = pd.DataFrame(data=self.counts, index=seqname, columns=columnanmes,dtype='double')
        # df.to_csv(self.outfile)
        return df


def remove_uncorrectchacter(infasta, savefasta):
    seqnames = []
    sequences = []
    dict_data = {}
    for seq in Seq.parse(infasta, 'fasta'):
        seqid = seq.id
        seqid = '>' + seqid
        n = 0
        sequence_o = str(seq.seq)
        sequence_o = sequence_o.upper()
        sequence_r = sequence_o.replace("U", "T")
        charterlist = ['B',  'D', 'E', 'F',  'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',  'U', 'V', 'W', 'X', 'Y', 'Z']
        for unchar in charterlist:
            # print(unchar)
            # print(type(unchar))
            sequence_r = sequence_r.replace(unchar, "")
        seqnames.append(seqid)
        sequences.append(sequence_r)
        dict_data[seqid] = sequence_r

    
    name_seq = dict(zip(seqnames, sequences))
    # 重新写入新的删除非法字符的fasta文件
    A_save = open(savefasta, 'w')
    for seqname in list(dict_data.keys()):
        a_seq = dict_data[seqname]
        A_save.write(str(seqname) + '\n' + str(a_seq) + '\n')
    A_save.close()


def construct_kmer():
    ntarr = ("A","C","G","T")

    kmerArray = []


    for n in range(4):
        kmerArray.append(ntarr[n])

    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            kmerArray.append(str2)
#############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                kmerArray.append(str3)
#############################################
#change this part for 3mer or 4mer
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    kmerArray.append(str4)
############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    for z in range(4):
                        str5 = str4 + ntarr[z]
                        kmerArray.append(str5)
####################### 6-mer ##############
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    for z in range(4):
                        str5 = str4 + ntarr[z]
                        for t in range(4):
                            str6 = str5 + ntarr[t]
                            kmerArray.append(str6)
####################### 7-mer ##############
    kmer7 = []
    for m in kmerArray[1364:5460]:
        for i in ntarr:
            st7 = m + i
            kmer7.append(st7)
    kmerArray = kmerArray+kmer7
    
    return kmerArray

def g_gap_single(seq,ggaparray,g):
    # seq length is fix =23
    length = len(seq)
    increment = 1000 / length 
    rst = np.zeros((16))
    for i in range(len(seq)-1-g):
        str1 = seq[i]
        str2 = seq[i+1+g]
        idx = ggaparray.index(str1+str2)
        rst[idx] += increment
    return rst

# binucleic ggap
# kmerarray[64:340]
def big_gap_single(seq,ggaparray,g):
    # seq length is fix =23
    datares = {}
    length = len(seq)
    increment = 1000 / length 
    rst = np.zeros((256))
    for i in range(len(seq)-3-g):
        str1 = seq[i]+seq[i+1]
        str2 = seq[i+2+g]+seq[i+2+g+1]
        idx = ggaparray.index(str1+str2)
        rst[idx] += increment
        datares[str1+str2] = rst[idx]

    return rst

def ggap_encode(seq,ggaparray,g):
    result = []
    for x in seq:
        temp = g_gap_single(x,ggaparray,g)
        result.append(temp)
    result = np.array(result)
    colnums = [*map(lambda x:str(g) + '_' + x, ggaparray)]
    df_res = pd.DataFrame(data = result, columns = colnums)
    print(df_res) 
    return df_res


def biggap_encode(seq,ggaparray,g):

    result = []
    for x in seq:
        temp = big_gap_single(x,ggaparray,g)
        result.append(temp)
    result = np.array(result)
    colnums = [*map(lambda x:str(g) + '_' + x, ggaparray)]
    df_res = pd.DataFrame(data = result, columns = colnums)
    print(df_res) 

    return df_res


def encode(records):

    RNA_seq = []
    RNA_id=[]
    for i in range(len(records)):
        RNA_seq.append(records[i].seq)   
        RNA_id.append(records[i].id)


    kmerArray = construct_kmer()
    for g in range(1,41,1):
        time0 = time.time()
        # print('g: {}'.format(g))
        RNA_ggap = ggap_encode(RNA_seq,kmerArray[4:20],g)
        if g==1:
            RNA_data = RNA_ggap
        else:
            RNA_data = pd.concat([RNA_data,RNA_ggap],axis=1)
        print('g time: {}: {} mins'.format(g, (time.time()-time0)/60))

    RNA_data.index = RNA_id
    return RNA_data

def calculate_gap_features(fastapath,output_path):

   
    datanm = fastapath.split('/')[-1].split('.')[0]
    savefasta =  datanm + '_01.fa'
    filename = datanm + '_gaps_'
    remove_uncorrectchacter(fastapath, savefasta)
    records = list(Seq.parse(savefasta, "fasta")) #  Human_orf_test_nc

    for k in range(1,4,1):
        print('k: {}'.format(k))
        Kmer = BasicCounter(savefasta, int(k)).get_counts()
        if k==1:
            RNA_kmer = Kmer
        else:
            RNA_kmer = pd.concat([RNA_kmer,Kmer],axis=1)
    print(RNA_kmer)

    RNA_gap = encode(records)
    print(RNA_gap.shape)
    print(RNA_gap)

    RNA_data = pd.merge(RNA_kmer,RNA_gap,left_index=True,right_index=True)
    print(RNA_data.shape)
    print(RNA_data)
    savefile = output_path + '/' + filename + str(RNA_data.shape[0]) + '_'+ str(RNA_data.shape[1]) + '.csv'
    RNA_data.to_csv((savefile))
    return savefile
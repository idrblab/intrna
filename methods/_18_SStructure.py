import os
import pandas as pd
import Bio.SeqIO as Seq
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import RNA

from tqdm import tqdm
tqdm.pandas(ascii=True)
# os.environ['R_HOME'] = "/usr/bin/R" #or whereever your R is installed

def makeSStructure(fastapath):

    robjects.r('''
                makeSStructure <- function(fastapath){
                  library(LncFinder)
                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  Seqs <- LncFinder::run_RNAfold(demo_DNA.seq, RNAfold.path = "RNAfold", parallel.cores = 2)
                  result_2 <- LncFinder::lnc_finder(Seqs, SS.features = TRUE, format = "SS", frequencies.file = "human", svm.model = "human", parallel.cores = -1)
                  res2 <- result_2[,c(14:21)]
                  res2[is.na(res2)] <- 0
                  return(res2)
                }''')
    sstruc = robjects.r['makeSStructure'](fastapath)
    sstruc.columns = ["SDMFE: Secondary structural minimum free energy", "SFPUS: Secondary structural UP frequency paired-unpaired", "SLDLD: Structural logarithm distance to lncRNA of acguD", "SLDPD: Structural logarithm distance to pcRNA of acguD", "SLDRD: Structural logarithm distance acguD ratio", "SLDLN: Structural logarithm distance to lncRNA of acguACGU", "SLDPN: Structural logarithm distance to pcRNA of acguACGU", "SLDRN: Structural logarithm distance acguACGU ratio"]
    return sstruc

def extract_SSfeatures(fastapath):
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    robjects.r('''
                extract_SSfeatures <- function(fastapath){
                  library(LncFinder)
                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  Seqs <- LncFinder::run_RNAfold(demo_DNA.seq, RNAfold.path = "RNAfold", parallel.cores = 2)
                  result_2 <- LncFinder::extract_features(Seqs, label = NULL, SS.features = TRUE,format = "SS", frequencies.file = "human", parallel.cores = -1)
                  res2 <- result_2[,c(12:19)]
                  return(res2)
                }''')
    sstruc = robjects.r['extract_SSfeatures'](fastapath)
    sstruc.columns = ["SLDLD: Structural logarithm distance to lncRNA of acguD", "SLDPD: Structural logarithm distance to pcRNA of acguD", "SLDRD: Structural logarithm distance acguD ratio", "SLDLN: Structural logarithm distance to lncRNA of acguACGU", "SLDPN: Structural logarithm distance to pcRNA of acguACGU", "SLDRN: Structural logarithm distance acguACGU ratio","SDMFE: Secondary structural minimum free energy", "SFPUS: Secondary structural UP frequency paired-unpaired"]
    return sstruc

# def makeEIIP(fastapath):
#     # import rpy2.robjects as robjects
#     # from rpy2.robjects import pandas2ri
#     # pandas2ri.activate()
#
#     robjects.r('''
#                 makeEIIP <- function(fastapath){
#                   library(LncFinder)
#                   demo_DNA.seq <- seqinr::read.fasta(fastapath)
#                   result_1 <- LncFinder::lnc_finder(demo_DNA.seq, SS.features = FALSE, frequencies.file = "human", svm.model = "human", parallel.cores = 1)
#                   res2 <- result_1[,c(8:13)]
#                   return(res2)
#                 }''')
#     sstruc = robjects.r['makeEIIP'](fastapath)
#     return sstruc

def makelogDis(fastapath):
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    robjects.r('''
                makelogDis <- function(fastapath){
                  library(LncFinder)
                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  result_1 <- LncFinder::lnc_finder(demo_DNA.seq, SS.features = FALSE, frequencies.file = "human", svm.model = "human", parallel.cores = -1)
                  res2 <- result_1[,c(5:7)]
                  return(res2)
                }''')
    sstruc = robjects.r['makelogDis'](fastapath)
    return sstruc

def makeEIIP(fastapath):
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    robjects.r('''
                makeEIIP <- function(fastapath){
                  library(LncFinder)
                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  result_1 <- compute_EIIP(
                                demo_DNA.seq,
                                label = NULL,
                                spectrum.percent = 0.1,
                                quantile.probs = seq(0, 1, 0.25)
                                )
                  return(result_1)
                }''')
    sstruc = robjects.r['makeEIIP'](fastapath)
    sstruc.columns = ['EipSP: Electron-ion interaction pseudopotential signal peak','EipAP: Electron-ion interaction pseudopotential average power','EiSNR: Electron-ion interaction pseudopotential signal/noise ratio','EiPS0: Electron-ion interaction pseudopotential spectrum 0','EiPS1: Electron-ion interaction pseudopotential spectrum 0.25','EiPS2: Electron-ion interaction pseudopotential spectrum 0.5','EiPS3: Electron-ion interaction pseudopotential spectrum 0.75','EiPS4: Electron-ion interaction pseudopotential spectrum 1']
    return sstruc

def makeEucDist(fastapath):
    robjects.r('''
                makeEucDist <- function(fastapath){
                  library(LncFinder)
                  cds.seq = seqinr::read.fasta('/public/home/wangyx/LncRNA/smallRNA/methods/Data/gencode.v34.pc_transcripts_test.fa')
                  lncRNA.seq = seqinr::read.fasta('/public/home/wangyx/LncRNA/smallRNA/methods/Data/gencode.v34.lncRNA_transcripts_test.fa')
                  referFreq <- make_referFreq(
                            cds.seq,
                            lncRNA.seq,
                            k = 6,
                            step = 1,
                            alphabet = c("a", "c", "g", "t"),
                            on.orf = TRUE,
                            ignore.illegal = TRUE
                            )
                  
                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  EucDis <- compute_EucDistance(
                                demo_DNA.seq,
                                label = NULL,
                                referFreq,
                                k = 6,
                                step = 1,
                                alphabet = c("a", "c", "g", "t"),
                                on.ORF = FALSE,
                                auto.full = FALSE,
                                parallel.cores = -1
                                )
                                
                                
                LogDistance <- compute_LogDistance(
                                    demo_DNA.seq,
                                    label = NULL,
                                    referFreq,
                                    k = 6,
                                    step = 1,
                                    alphabet = c("a", "c", "g", "t"),
                                    on.ORF = FALSE,
                                    auto.full = FALSE,
                                    parallel.cores = -1
                                    )
                hdata2<-cbind(EucDis,LogDistance)
                  return(hdata2)
                }''')
    sstruc = robjects.r['makeEucDist'](fastapath)
    return sstruc

def makeORFEucDist(fastapath):
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()
    r_script = '''
                makeORFEucDist <- function(fastapath){
                  library(LncFinder)
                  cds.seq = seqinr::read.fasta('/public/home/wangyx/LncRNA/smallRNA/methods/Data/gencode.v34.pc_transcripts_test.fa')
                  lncRNA.seq = seqinr::read.fasta('/public/home/wangyx/LncRNA/smallRNA/methods/Data/gencode.v34.lncRNA_transcripts_test.fa')
                  referFreq <- make_referFreq(
                            cds.seq,
                            lncRNA.seq,
                            k = 6,
                            step = 1,
                            alphabet = c("a", "c", "g", "t"),
                            on.orf = TRUE,
                            ignore.illegal = TRUE
                            )

                  demo_DNA.seq <- seqinr::read.fasta(fastapath)
                  EucDis <- compute_EucDistance(
                                demo_DNA.seq,
                                label = NULL,
                                referFreq,
                                k = 6,
                                step = 1,
                                alphabet = c("a", "c", "g", "t"),
                                on.ORF = TRUE,
                                auto.full = FALSE,
                                parallel.cores = -1
                                )


                LogDistance <- compute_LogDistance(
                                    demo_DNA.seq,
                                    label = NULL,
                                    referFreq,
                                    k = 6,
                                    step = 1,
                                    alphabet = c("a", "c", "g", "t"),
                                    on.ORF = TRUE,
                                    auto.full = FALSE,
                                    parallel.cores = -1
                                    )
                hexamerScore <- compute_hexamerScore(
                                    demo_DNA.seq,
                                    label = NULL,
                                    referFreq,
                                    k = 6,
                                    step = 1,
                                    alphabet = c("a", "c", "g", "t"),
                                    on.ORF = TRUE,
                                    auto.full = FALSE,
                                    parallel.cores = -1
                                    )
                hdata2<-cbind(EucDis,LogDistance)
                result <- cbind(hdata2,hexamerScore)
                  return(result)
                }
                '''
    robjects.r(r_script)
    sstruc = robjects.r['makeORFEucDist'](fastapath)
    sstruc.columns = ['EucDist.LNC_orf', 'EucDist.PCT_orf', 'EucDist.Ratio_orf', 'LogDist.LNC_orf', 'LogDist.PCT_orf', 'LogDist.Ratio_orf', 'Hexamer.Score_orf']
    return sstruc

def cal_base_pairs(fasta):
  # print('mfe is')
  nMFEs = []
  base_pairs = []
  all_struct = []
  seqname = []
  num = 0
  # for seq in tqdm(Seq.parse(fasta,'fasta'), ascii = True):
  for seq in Seq.parse(fasta,'fasta'):
    num += 1
    print('the {}: seq.id is {}'.format(num, seq.id))
    # print('seq.seq is {}'.format(seq.seq))

    RNA_size = len(seq.seq)
    if RNA_size > 30000:
        sequence_o = str(seq.seq[0:30000])
    else:
        sequence_o = str(seq.seq)
    print(len(sequence_o))
    (ss, mfe) = RNA.fold(sequence_o)
    # mfe = 0
    # ss = '...(('
    print('mfe is {}'.format(mfe))
    MFEs = mfe
    seq_len = len(sequence_o)
    nMFEs.append(float(MFEs)/seq_len)

    base_pairs.append(ss.count('('))

    seqid = seq.id
    seqname.append(seqid)
 
    struct = ss
    struct = struct.replace(')', '(')
    # print('struct')
    # print(struct)

    # list and position map for k_mer structure
    k_mers = ['']
    k_mer_struct_list = []
    k_mer_struct_map = {}
    structs = '.('
    for T in range(4):
        temp_list = []
        for k_mer in k_mers:
            for s in structs:
              temp_list.append(k_mer + s)
        k_mers = temp_list
        k_mer_struct_list += temp_list
    for i in range(len(k_mer_struct_list)):
        k_mer_struct_map[k_mer_struct_list[i]] = i

    result_struct = []
    offset_struct = 0
    for K in range(1, 4 + 1):
      vec_struct = [0.0] * (2 ** K)
      counter = seq_len - K + 1
      for i in range(seq_len - K + 1):
          k_mer_struct = struct[i:i + K]
          # print('seq_len is {}'.format(seq_len))
          vec_struct[k_mer_struct_map[k_mer_struct] - offset_struct] += 1 # vec_struct是所有元素的频数
      vec_struct = np.array(vec_struct)
      offset_struct += vec_struct.size                    
      vec_struct = vec_struct / vec_struct.max()
      result_struct += list(vec_struct)
      # print('result_struct')
      # print(result_struct)
    all_struct.append(result_struct)

  # print(len(seqname))
  all_struct_df = pd.DataFrame(all_struct, columns = k_mer_struct_list, index=seqname)
  # all_struct_df = pd.DataFrame(all_struct, columns = k_mer_struct_list)
  # print(all_struct_df)
  dict_data = {
      'nMFE': nMFEs,
      'Base pairs' : base_pairs
  }
  df = pd.DataFrame(dict_data,index=seqname)
  # df = pd.DataFrame(dict_data)
  fina_df = pd.concat([df,all_struct_df],axis = 1)
  # print(seqname)

  print('fina_df.head()')
  print(fina_df.head())

  return fina_df
    

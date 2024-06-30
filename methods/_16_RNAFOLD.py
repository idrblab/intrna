import os
import subprocess
import numpy as np
import shutil

vocab_dict = {'S': [1, 0, 0, 0, 0, 0, 0],
              'I': [0, 1, 0, 0, 0, 0, 0],
              'H': [0, 0, 1, 0, 0, 0, 0],
              'E': [0, 0, 0, 1, 0, 0, 0],
              'M': [0, 0, 0, 0, 1, 0, 0],
              'B': [0, 0, 0, 0, 0, 1, 0],
              'X': [0, 0, 0, 0, 0, 0, 1]}
def rnaflod(path, path2):
    os.chdir(path2)
    subprocess.call(["/home/wangyunxia/downloadApp/RNAfold", '--noPS', '-i', path, '-o'])
    # subprocess.call(['RNAfold', '--help'])
def process(filepath, savepath):
    listfile = os.listdir(filepath)
    for eachfile in listfile:
        if os.path.splitext(eachfile)[1] == '.fold':

            with open(filepath + '/' + eachfile, 'r') as f:

                t = 0
                for line in f:
                    if t == 0:
                        name = line.strip('\n').split('|')[0]
                        name1 = name.strip('>')
                    if t == 1:
                        seq = line.strip('\n')
                    if t == 2:
                        secondary_s = line.strip('\n').split()[0]
                    t += 1
                file = open(savepath + '/' + name1 + '.dbn', 'w')
                file.write(name + '\n' + seq + '\n' + secondary_s)
                file.close()

def second_file(filepath):
    file = os.listdir(filepath)
    for eachfile in file:
        if os.path.splitext(eachfile)[1] == '.dbn':
            arg = filepath + '/' + eachfile

            subprocess.call(["/usr/bin/perl", "/home/wangyunxia/RNACode/tornadoBulid/LncRNAcoder/methods/_1601_bprna.pl", arg])
def encode_feature(filepath, N):
    file = os.listdir(filepath)
    encoded_all = []
    seqname = []

    for eachfile in file:

        if os.path.splitext(eachfile)[1] == '.st':
            arg = filepath + '/' + eachfile

            with open(arg, 'r') as f:
                t = 0
                for line in f:

                    if t == 5:
                        line = line.strip('\n')

                        ee = []


                        for aa in line:
                            encoded = vocab_dict[aa]
                            ee.append(encoded)
                        break
                    t += 1
            ee = np.array(ee)
            new_array = np.zeros((N, 7))
            if len(ee) >= N:
                new_array[:, :] = ee[0:N, :]
            if len(ee) < N:
                new_array[0:len(ee), :] = ee

            namefile = os.path.splitext(eachfile)[0]
            seqname.append(namefile)
            encoded_all.append(new_array)

    encoded_all = np.array(encoded_all)

    return seqname,encoded_all


def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
if __name__ == '__main__':
    path = '/home/wangyunxia/RNACode/tornadoBulid/LncRNAcoder/statics/data/UploadSampleData-test/RNA-RNA/SampleData-miRNA-B.fasta'
    path2 = '/home/wangyunxia/RNACode/tornadoBulid/LncRNAcoder/methods/RNAFoldOut'
    setDir(path2)
    rnaflod(path, path2)
    process(path2, path2)
    second_file(path2)
    seqname,encoded_all = encode_feature(path2,1000)
    print(seqname)
    # encoded_all = np.array(encoded_all)
    print(encoded_all.shape)

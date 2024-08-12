import numpy as np
import scipy as scp
import itertools as it
import copy
import time
import datetime
import sys
import os
import pathlib
import shutil
import subprocess
import concurrent.futures
from pyscf import gto
from pyscf.tools import molden

@np.vectorize
def mapToString(a):
    if a == 1.:
        return 'a'
    elif a == 1.j:
        return 'b'
    elif a == (1. + 1.j):
        return 'd'
    else:
        return 'e'

def writeWfun(wfun,ciVec,destDir,suffix):
    nStates = ciVec.shape[1]
    nMO = len(list(wfun.keys())[0])
    nDets = len(wfun.keys())
    header = f'{nStates:d} {nMO:d} {nDets:d}\n' 
    content = ''
    for det, infoDet in wfun.items():
        ind, phase = infoDet
        ciCoeff = ciVec[ind] * phase
        content += det + ' '
        content += ' '.join(f'{x:11.7f}' for x in ciCoeff.tolist())
        content += '\n'
    detFileName = os.path.join(destDir,'det.' + suffix)
    with open(detFileName, "w") as detFile:
        detFile.write(header)
        detFile.write(content)

def getXMSrot(xmsFileName):
    xmsFileLines = (line for line in open(xmsFileName,'r'))
    xmsLines=it.takewhile(
        lambda x: len(x) > 1,
        it.filterfalse(
            lambda x: '* XMS-CASPT2 rotation matrix' in x,
            it.dropwhile(
                lambda x: not '* XMS-CASPT2 rotation matrix' in x,
                xmsFileLines
            )
        )
    )
    xmsRot = [[float(x) for x in xmsLine.strip().split()] for xmsLine in xmsLines]
    return np.array(xmsRot)

def writeAOovl(destDir, aoS):
    dim1, dim2 = aoS.shape
    header = f'{dim1:d} {dim2:d}\n' 
    content = ''
    for row in aoS:
        for element in row:
            content += f'{element: 9.7e} '
        content += '\n'

    destFile = os.path.join(destDir, 'aoovl') 
    with open(destFile, 'w') as AOovlFile:
        AOovlFile.write(header)
        AOovlFile.write(content)

def writeMOFile(destDir, mo, suffix):
    dim1, dim2 = mo.shape
    header = '2mocoef\nheader\n 1\nMO-coeff\n 1\n' 
    header += f'{dim1:4d}  {dim2:4d}\n' 
    header += ' a\nmocoef\n(*)\n' 
    content = ''
    count = 1
    for element in np.ravel(mo.T):
        content += f'{element: 14.12e} '
        count += 1
        if count > dim1: 
            count = 1
            content += '\n'
        

    destFile = os.path.join(destDir, 'mo.' + suffix) 
    with open(destFile, 'w') as MOFile:
        MOFile.write(header)
        MOFile.write(content)

def writeWFovlInp(destDir):
    content = r'''mix_aoovl=aoovl
a_mo=mo.1
b_mo=mo.2
a_det=det.1
b_det=det.2
a_mo_read=0
b_mo_read=0
ao_read=0
moprint=2
'''
    destFile = os.path.join(destDir, 'wfovl.inp') 
    with open(destFile, "w") as wfOvlInp:
        wfOvlInp.write(content)

def joinDets(alpha, beta, ind):
    dets = {}
    aMask = np.any(alpha==-1,axis=1)
    bMask = np.any(beta==-1,axis=1)
    tmp = alpha + 1.j * beta 
#    detStrings = mapToString(tmp)
    for ia, a in enumerate(alpha):
        b = beta[ia]
        if aMask[ia] or bMask[ia]:
            continue
        detString = ''.join(mapToString(tmp[ia]))
        a = np.sum(tmp[ia][:ind])
        ninv = np.real(a) + np.imag(a)
        if detString[ind] == 'a':
            ninv += 1 
        phase = (-1)**ninv
#        phase = 1
        #detString2 = ''
        dets[detString] = (ia, phase)

    return dets

class DetAnnihilator:
    def __init__(self,alpha, beta, nMO):
        self.alpha = alpha
        self.beta = beta
        self.nMO = nMO
        self.identity = np.eye(nMO)

    def __call__(self,iMO):
        annihilatedColumn = self.identity[iMO]
        annihilatedAlpha = self.alpha.astype(int) - annihilatedColumn 
        annihilatedBeta = self.alpha.astype(int) - annihilatedColumn 
        annihilatedAlphaDet = joinDets(annihilatedAlpha,self.beta,iMO)
        annihilatedBetaDet = joinDets(self.alpha,annihilatedBeta,iMO)
        return iMO,annihilatedAlphaDet, annihilatedBetaDet
        

def annihilateDets(alpha, beta, nMO):
    annihilateDet = DetAnnihilator(alpha, beta, len(alpha[0])) 
    annihilatedDetStrings = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for moData in executor.map(annihilateDet,range(nMO)):
            iMO, alpha, beta = moData
            print(f'{iMO} {datetime.datetime.now()}' )
            annihilatedDetStrings[iMO] = {'alpha': alpha, 'beta': beta}
    return annihilatedDetStrings
    #alphaA = self.alpha.astype(int) - annihilatorA 
    #alphaB = self.alpha.astype(int) - annihilatorB 
    #betaA = self.beta.astype(int) - annihilatorA 
    #betaB = self.beta.astype(int) - annihilatorB 

class Annihilator:
    def __init__(self, alphaA, betaA, alphaB, betaB, ciVecsA, ciVecsB, nMO, work):
        self.alphaA = alphaA
        self.betaA = betaA
        self.alphaB = alphaB
        self.betaB = betaB
        self.ciVecsA = ciVecsA
        self.ciVecsB = ciVecsB
        self.nStatesA = ciVecsA.shape[1] 
        self.nStatesB = ciVecsB.shape[1]
        self.work = work
        self.wfovlExe = os.environ["WFOVERLAP"]
        self.annihilatedDetsA = annihilateDets(alphaA, betaA, nMO)
        self.annihilatedDetsB = annihilateDets(alphaB, betaB, nMO)

    def __call__(self, moPair):
        start = time.time()
        iMO, jMO = moPair 
        commonDir, alphaDir, betaDir = self._setupDirs(iMO, jMO)
        matAlpha = self._executeWFOvl(alphaDir)
        matBeta = self._executeWFOvl(betaDir)
        #mat = matAlpha + matBeta
        #print(iMO,jMO)
        self._cleanDirs(commonDir)
        end = time.time()
        #print(end-start)
        return iMO, jMO, matAlpha, matBeta, end-start

    def _setupDirs(self, iMO, jMO):
        commonDir = os.path.join(self.work, f'{iMO:d}_{jMO:d}') 
        alphaDir = os.path.join(commonDir, 'alpha') 
        betaDir = os.path.join(commonDir, 'beta') 
        os.mkdir(commonDir)
        os.mkdir(alphaDir)
        os.mkdir(betaDir)
#        nMO = len(self.alpha[0])
#        identity = np.eye(nMO)
#        annihilatorA = identity[iMO]
#        annihilatorB = identity[jMO]
#        alphaA = self.alpha.astype(int) - annihilatorA 
#        alphaB = self.alpha.astype(int) - annihilatorB 
#        betaA = self.beta.astype(int) - annihilatorA 
#        betaB = self.beta.astype(int) - annihilatorB 
#        alphaDets = [alphaA,self.alpha,alphaB,self.alpha]
#        betaDets = [self.beta,betaA,self.beta,betaB]
#        dirs = [alphaDir,betaDir,alphaDir,betaDir]
#        suffixes = ['1','1','2','2']
        #start = time.time()
#        annihilatedAlphaA = joinDets(alphaA,self.beta,iMO)
#        annihilatedBetaA = joinDets(self.alpha,betaA,iMO)
        #end = time.time()
        #print(end-start)
#        annihilatedAlphaB = joinDets(alphaB,self.beta,jMO)
#        annihilatedBetaB = joinDets(self.alpha,betaB,jMO)
        writeWfun(self.annihilatedDetsA[iMO]['alpha'],self.ciVecsA,alphaDir,'1')
        writeWfun(self.annihilatedDetsB[jMO]['alpha'],self.ciVecsB,alphaDir,'2')
        writeWfun(self.annihilatedDetsA[iMO]['beta'],self.ciVecsA,betaDir,'1')
        writeWfun(self.annihilatedDetsB[jMO]['beta'],self.ciVecsB,betaDir,'2')
        #print(end-start)
        self._copyFiles(alphaDir)
        self._copyFiles(betaDir)
        return commonDir, alphaDir, betaDir

    def _copyFiles(self, destDir):
        srcMoA = os.path.join(self.work, 'mo.1') 
        srcMoB = os.path.join(self.work, 'mo.2') 
        srcAOovl = os.path.join(self.work, 'aoovl') 
        srcWFovl = os.path.join(self.work, 'wfovl.inp') 
        destMoA = os.path.join(destDir, 'mo.1') 
        destMoB = os.path.join(destDir, 'mo.2') 
        destAOovl = os.path.join(destDir, 'aoovl') 
        destWFovl = os.path.join(destDir, 'wfovl.inp') 
        shutil.copyfile(srcMoA,destMoA)
        shutil.copyfile(srcMoB,destMoB)
        shutil.copyfile(srcAOovl,destAOovl)
        shutil.copyfile(srcWFovl,destWFovl)

    def _executeWFOvl(self, destDir):
        os.chdir(destDir)
        wfovlProc = subprocess.run([self.wfovlExe, '-f', 'wfovl.inp'], capture_output=True, encoding='utf-8')
        output = getSection("Overlap matrix","Renormalized",wfovlProc.stdout.split('\n'))
        mat = self._processMat(output)
        return mat
        #for line in output:
        #    print(line.strip())

    def _processMat(self, output):
        mat = np.zeros((self.nStatesA, self.nStatesB))
        headLines = 2
        headLine = 1
        read = False
        for line in output:
            if not(read) and (headLine == headLines):
                read = True
                iRow = 0
                continue

            if read:
                #print(line)
                mat[iRow,:] = np.array([float(x) for x in line.strip().split()[2:]])
                iRow += 1
                if iRow == self.nStatesA:
                    break
                continue
        
            headLine += 1
        return mat

    def _cleanDirs(self, commonDir):
        shutil.rmtree(commonDir)

def calculateTDen(alpha, beta, moA, moB, ciVecsA, ciVecsB, aoS, nMO, moS): 
    cwd = os.getcwd()
    print(cwd)
    work = os.path.join(cwd, 'work')
    if os.path.exists(work):
        shutil.rmtree(work)
    os.mkdir(work)
    writeAOovl(work, aoS)
    writeMOFile(work, moA, '1')
    writeMOFile(work, moA, '2')
    writeWFovlInp(work)
    nStatesA = ciVecsA.shape[1]
    nStatesB = ciVecsB.shape[1]
    moInds = range(nMO)
    moPairs = list(it.product(moInds,moInds))
    tDenAlpha = np.zeros((nStatesA, nStatesB, nMO, nMO)) 
    tDenBeta = np.zeros((nStatesA, nStatesB, nMO, nMO)) 
    annihilate = Annihilator(alpha, beta, alpha, beta, ciVecsA, ciVecsB, nMO, work)
    tottime = 0 
    n = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_pair = (executor.submit(annihilate, moPair) for moPair in moPairs)
        for future in concurrent.futures.as_completed(future_pair):
            iMO, jMO, matAlpha, matBeta, time = future.result() 
            tottime += time
            n += 1
            print(f'Annihilated MOs: {iMO:d}-{jMO:d} at {datetime.datetime.now()}')
            tDenAlpha[:,:,iMO,jMO] = matAlpha
            tDenBeta[:,:,iMO,jMO] = matBeta
            print(tottime, n, tottime/n)
        #for pairData in executor.map(annihilate, moPairs):
        #    iMO, jMO, matAlpha, matBeta, time = pairData 
        #    tottime += time
        #    n += 1
        #    print(f'Annihilating MOs: {iMO:d}-{jMO:d} at {datetime.datetime.now()}')
        #    tDenAlpha[:,:,iMO,jMO] = matAlpha
        #    tDenBeta[:,:,iMO,jMO] = matBeta
        #    print(tottime, n, tottime/n)
    os.chdir(cwd)
    return tDenAlpha, tDenBeta

def getSection(start, end, inputGenerator):
    dropCond = lambda x: not start in x
    takeCond = lambda x: not end in x
    head = it.takewhile(takeCond, inputGenerator) 
    return it.dropwhile(dropCond, head)

def splitLines(splitCond, inputGenerator, ignoreCond=None):
    outputChunk = []
    for line in inputGenerator:
        if splitCond(line):
            yield outputChunk
            outputChunk = []
            continue
        if (ignoreCond != None) and ignoreCond(line): 
            continue
        outputChunk.append(line.strip().split())

def getCIvectors(inputFile):
    fileLines = (line for line in open(inputFile,'r'))
    ciVecLines = getSection('* ci vector', '* METHOD: CASSCF', fileLines)
    ignoreCond = lambda x: '* ci vector' in x
    splitCond = lambda x: len(x) == 1 
    ciVectors = splitLines(splitCond,ciVecLines,ignoreCond=ignoreCond) 
    detInds = {}
    ciCoeffs = []
    for iCiVector, ciVector in enumerate(ciVectors): 
        ciCoeff = []
        for iDet, det in enumerate(ciVector):
            detString = det[0]
            ciCoeff.append(float(det[1]))
            if detString not in detInds:
                detInds[detString] = [(iCiVector, iDet)] 
            else:
                detInds[detString].append((iCiVector, iDet))
        ciCoeffs.append(ciCoeff)

    ciVecs = np.zeros((len(detInds),iCiVector+1)) 
    for iDetString, detString in enumerate(detInds):
        for iDetInd, detInd in enumerate(detInds[detString]):
            ciVecs[iDetString, detInd[0]] = ciCoeffs[detInd[0]][detInd[1]]
   
    return list(detInds.keys()), ciVecs

def getOrbitalNrs(inputFile):
    fileLines = (line for line in open(inputFile,'r'))
    orbitalNrLines = getSection('* nclosed', 'CASSCF iteration', fileLines)
    nrOrbs = ()
    for orbitalNrLine in orbitalNrLines:
        nrOrbs += (int(orbitalNrLine.strip().split()[-1]),)
    return nrOrbs 

def peek(inputGenerator):
    original, new = it.tee(inputGenerator,2)
    element = next(new)
    return original, element

def getSpinSeparatedDets(inputDetStrings):
    nalpha = sum(1 if (x == '2') or (x == 'a')
                 else 0 for x in inputDetStrings[0])
    nOrbs = len(inputDetStrings[0])
    nDets = len(inputDetStrings)
    alphaDets = np.zeros((nDets, nOrbs))
    betaDets = np.zeros((nDets, nOrbs))
    inputDetStrings = np.array(inputDetStrings, dtype=str)
    inputDetStrings = inputDetStrings.view('<U1').reshape((inputDetStrings.size,-1))
    start = time.time()
    aMask = (inputDetStrings == '2') + (inputDetStrings == 'a')
    bMask = (inputDetStrings == '2') + (inputDetStrings == 'b')
    accAlpha = np.add.accumulate(aMask.astype(int), axis=0)

    accInv = nalpha - accAlpha 
    return aMask, bMask 

def padDetStrings(nclosed, nactive, nvirtual, inputDetStrings):
    startString = "2"*nclosed
    endString = "."*nvirtual
    outputDetStrings = []
    for detString in inputDetStrings:
        paddedDetString = startString + detString + endString
        outputDetStrings.append(paddedDetString)
    return outputDetStrings

def writeTDen(tDenAlpha,tDenBeta,cwd):
    nstate1, nstate2, nMO, nMO = tDenAlpha.shape 
    for i in range(nstate1):  
        for j in range(nstate2):  
            np.savetxt(pathlib.PurePath(cwd,f'tdenAlpha{i:d}-{j:d}.dat'), tDenAlpha[i,j,:,:], fmt='% 14.12e')
            np.savetxt(pathlib.PurePath(cwd,f'tdenBeta{i:d}-{j:d}.dat'), tDenBeta[i,j,:,:], fmt='% 14.12e')

if __name__ == "__main__":
    #moldena = molden.load('./orbitals.molden.2')
    moldenb = molden.load('./orbitals.molden.1')
    #mola = moldena[0]
    #Ca = moldena[2]
    molb = moldenb[0]
    Cb = moldenb[2]
    #
    #aoSab = gto.mole.intor_cross('int1e_ovlp_sph',mola,molb) 
    #moSab = np.matmul(Ca.T,np.matmul(aoSab,Cb))
    aoS = molb.intor('int1e_ovlp_sph') 
    moS = np.matmul(Cb.T,np.matmul(aoS,Cb))
    start=time.time()
    cwd = os.getcwd()
    detStrings, ciVecs = getCIvectors('./BAGEL_1.out')
    nclosed, nactive, nvirtual = getOrbitalNrs('./BAGEL_1.out')
    detStrings = padDetStrings(nclosed, nactive, nvirtual, detStrings)
    alphaDet, betaDet = getSpinSeparatedDets(detStrings)
    xmsrot = getXMSrot('./BAGEL_1.out')
    ciVecs = np.matmul(ciVecs,xmsrot)
    tDenAlpha, tDenBeta = calculateTDen(alphaDet, betaDet, Cb, Cb, ciVecs, ciVecs, aoS, nclosed + nactive, moS)
    writeTDen(tDenAlpha, tDenBeta, cwd)
    end=time.time()
    diff = end - start
    print(diff)

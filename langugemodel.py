from nltk.corpus import stopwords
from collections import Counter
import re, string
import math, random
import operator, heapq, functools

N = 1024908267229 ## Number of tokens

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as file:
        for line in file: yield line.split(sep)
    file.close()

def bigrams(name, sep='\t'):
    __prev = ""
    "Read key,value pairs from file."
    with open(name) as file:
        for line in file:
                for word in line.split():
                    yield (__prev, word)
                    __prev = word
                __prev = "" 
    file.close()           

def unknown_wrd(key, N):
    "Estimate the probability of an unknown word."
    return 10./(N * 10**len(key))

class LanguageModel(object):
    "Language model based on vocabulary(data, N) and probability to see an unknown word (missingfn)"
    def __init__(self, data=[], N=None, missingfn=None):
        for __wrd in data: 
            if not __wrd.isdigit(): self[__wrd] = self.get(__wrd, 0) + 1
            else: pass
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  #P(w)
        else: return self.missingfn(key, self.N) #P(unknown)

class ngramModel(object):
    "Collocation model)"
    def __init__(self, data=[], N=None):
        for (__prev, __key) in data: 
            if (not __key.isdigit()) or (not __prev.isdigit()): self[(__prev,__key)] = self.get((__prev,__key), 0) + 1
            else: pass
        self.N = float(N or sum(self.itervalues()))
    def __call__(self, __key, __prev): 
        if key in self: return self[(__prev, __key)]/ugrm(__prev)  #P(w,c)/P(w)
 
class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

for i in range(10):
    ugrm = LanguageModel(datafile('./pa2-data/data/corpus/'+ str(i) + '.txt'), N, unknown_wrd(1,N))

for i in range(10):
    bigram = ngramModel(bigrams('./pa2-data/data/corpus/'+ str(i) + '.txt'), N)
    
def compute_intp(key, prev):
    "Compute interpolated probability"
    intp = lmbd * ugrm(key) + (1-lmbd) * ngramModel(key, prev)
    return intp
    
def corrections(text): 
    "Spell-correct all words in text." 
    for line in text: words = line.split()
    prev = None
    for w in words: 
        return correct(w, prev)
        prev = w

def correct(w,p): 
    "Return the word that is the most likely spell correction of w." 
    candidates = edits(w).items() 
    c, edit = max(candidates, key=lambda (c,e): log10(Pedit(e) * bigrams(c,p)) 
    return c 

def Pedit(edit): 
    "The probability of an edit; can be '' or 'a|b' or 'a|b+c|d'." 
    if edit == '': return (1. - p_spell_error) 
    return p_spell_error*product(P1edit(e) for e in edit.split('+')) 

p_spell_error = 1./20. 

P1edit = Pdist(datafile('count_1edit.txt')) ## Probabilities of single edits 

def edits(word, d=2): 
    "Return a dict of {correct: edit} pairs within d edits of word." 
    results = {} 
    def editsR(hd, tl, d, edits): 
        def ed(L,R): return edits+[R+'|'+L] 
        C = hd+tl 
        if C in ugrm: 
            e = '+'.join(edits) 
            if C not in results: results[C] = e 
            else: results[C] = max(results[C], e, key=Pedit) 
        if d <= 0: return 
        extensions = [hd+c for c in alphabet if hd+c in PREFIXES] 
        p = (hd[-1] if hd else '<') ## previous character 
        ## Insertion 
        for h in extensions: 
            editsR(h, tl, d-1, ed(p+h[-1], p)) 
        if not tl: return 
        ## Deletion 
        editsR(hd, tl[1:], d-1, ed(p, p+tl[0])) 
        for h in extensions: 
            if h[-1] == tl[0]: ## Match 
                editsR(h, tl[1:], d, edits) 
            else: ## Replacement 
                editsR(h, tl[1:], d-1, ed(h[-1], tl[0])) 
        ## Transpose 
        if len(tl)>=2 and tl[0]!=tl[1] and hd+tl[1] in PREFIXES: 
            editsR(hd+tl[1], tl[0]+tl[2:], d-1, 
                   ed(tl[1]+tl[0], tl[0:2])) 
    ## Body of edits: 
    editsR('', word, d, []) 
    return results 

PREFIXES = set(w[:i] for w in ugrm for i in range(len(w) + 1)) 
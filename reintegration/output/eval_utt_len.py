import pickle, numpy as np
d = pickle.load(open('feature/audio/mfcc/meld/test.pkl','rb'))
short = {k:v.shape for k,v in d.items() if v.shape[0] < 8}
print(short)
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

raw_data = np.loadtxt('data/3.txt')
# print raw_data

train_obser_seq1 = (np.array([raw_data[:,0]]).T)[300:1100]  # sensor 1 angle
train_obser_seq2 = (np.array([raw_data[:,1]]).T)[300:1100]   # sensor 2 angle
train_obser_seq3 = (np.array([raw_data[:,4]]).T)[300:1100]   # sensor 3 angle
train_obser_seq4 = (np.array([raw_data[:,2]]).T)[300:1100]   # sensor 1 velocity
train_obser_seq5 = (np.array([raw_data[:,3]]).T)[300:1100]   # sensor 2 velocity
train_obser_seq6 = (np.array([raw_data[:,5]]).T)[300:1100]  # senosr 3 velocity
train_obser_seq = np.concatenate([train_obser_seq1, train_obser_seq2, train_obser_seq3,
                                  train_obser_seq4, train_obser_seq5, train_obser_seq6])
# train_obser_seq = train_obser_seq1
lengths_of_train = [len(train_obser_seq1), len(train_obser_seq2), len(train_obser_seq3),
                    len(train_obser_seq4), len(train_obser_seq5), len(train_obser_seq6)]
validation_obser_seq2 = (np.array([raw_data[:,0]]).T)[700:1100]

# plt.figure()
# plt.plot(train_obser_seq2)
plt.figure()
plt.plot(validation_obser_seq2)

print "==="

h = hmm.GaussianHMM(n_components=6, covariance_type="full")
h.fit(train_obser_seq, lengths_of_train)
# h.fit(train_obser_seq)

pi = h.startprob_
print pi
A = h.transmat_
print A
Bm = h.means_
print Bm
Bc = h.covars_
print Bc

validation_estimate_seq = h.predict(validation_obser_seq2)

tmp = []
seq = []
j = 0
for i in range(0, len(validation_estimate_seq)-1):
    if validation_estimate_seq[i] != validation_estimate_seq[i+1]:
        tmp.append(validation_estimate_seq[i])
        if len(tmp) > 6:
            if tmp[-1] == tmp[-1-6]:
                seq.append(tmp[-1-6])
                if len(seq) >= 6:
                    print "get seq"
                    break
            else:
                seq = []

print "seq: ", seq
print "tmp: ", tmp
estimate_states = []
for r in range(0, len(validation_estimate_seq)):
    estimate_states.append(seq.index(validation_estimate_seq[r]))

print estimate_states
plt.figure()
plt.plot(estimate_states)



print "validation estimate seq: ", validation_estimate_seq
plt.figure()
plt.plot(validation_estimate_seq)


plt.show()
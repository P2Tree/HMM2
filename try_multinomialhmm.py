import numpy as np
from hmmlearn import hmm
import xlrd
import matplotlib.pyplot as plt

# data = xlrd.open_workbook('data/walk.xlsx')
# table = data.sheet_by_name(u'Sheet1')
raw_data = np.loadtxt('data/result.txt')
# print raw_data
data = np.array([np.int32(raw_data)]).T
# print data
# plt.figure()
# plt.plot(data)
train_obser_seq2 = data[700:1100]
# print train_obser_seq2
# plt.figure()
# plt.plot(train_obser_seq2)
validation_obser_seq2 = data[300:700]
# print validation_obser_seq2
plt.figure()
plt.plot(validation_obser_seq2)

h = hmm.MultinomialHMM(n_components=6, n_iter=100)
h.fit(train_obser_seq2)

pi = h.startprob_
print pi
A = h.transmat_
print A
B = h.emissionprob_
print B

validation_estimate_seq = h.predict(validation_obser_seq2)

# print "validation_states_seq: ", validation_states_seq2
print "validation_estimate_seq: ", validation_estimate_seq

# p = 0.0
# for s in range(0, len(validation_estimate_seq.T)):
#     if validation_states_seq2.T[s] == validation_estimate_seq.T[s]:
#         p += 1
# print "Accuracy of validate: ", p / len(validation_estimate_seq.T)
plt.figure()
plt.plot(validation_estimate_seq)

print "==="

# test_estimate_seq = h.predict(test_obser_seq2)
#
# print "test_states_seq: ", test_states_seq2
# print "test_estimate_seq: ", test_estimate_seq
#
# p = 0.0
# for s in range(0, len(test_estimate_seq.T)):
#     if test_states_seq2.T[s] == test_estimate_seq.T[s]:
#         p += 1
#
# print "Accuracy of test: ", p / len(test_estimate_seq)




plt.show()

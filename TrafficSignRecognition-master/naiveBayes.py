from sklearn.naive_bayes import GaussianNB
import numpy
import time
gnb = GaussianNB()
feature_vector  = numpy.genfromtxt('feature_vector.csv', delimiter=',', dtype=float)
test_vector     = numpy.genfromtxt('feature_vector_test.csv', delimiter=',', dtype=float)
startTime		= time.time()
y_pred = gnb.fit(feature_vector[:,1:], feature_vector[:,:1].ravel())
training_time	= time.time()-startTime
print(training_time)
y_pred = y_pred.predict(test_vector[:,1:])
print(y_pred.astype(int),test_vector[:,:1].ravel().astype(int),test_vector.shape[0])
print((y_pred.astype(int)==test_vector[:,:1].ravel().astype(int)).sum()/test_vector.shape[0])

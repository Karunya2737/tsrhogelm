from hpelm.elm import ELM
#import elm as ELM
import numpy
class elm():

    def train(self,features,nnc1,nnc2):

        X   = features[:,1:]
        T   = features[:,:1]
        T   = T.ravel()
        print(T)
        list2   = []
        list1   = [0 for _ in range(43)]
        for i in range(X.shape[0]):
            temp    = list(list1)
            temp[int(T[i])]    = 1
            list2.append(temp)
        T   = numpy.array(list2)
        elmk = ELM(X.shape[1],43)
        elmk.add_neurons(nnc1, "tanh")
        elmk.add_neurons(nnc2, "sigm")
        elmk.train(X, T, "LOO",'c', batch = 1000)
        Y   = elmk.predict(X)
        print(Y)
        elmk.save('model')

    def test(self,features):
        X   = features[:,1:]
        T   = features[:,:1]
        T   = T.astype(int)
        print(T)
        elmk = ELM(X.shape[1],43)
        elmk.load('model')
        output  = elmk.predict(X)
        return output

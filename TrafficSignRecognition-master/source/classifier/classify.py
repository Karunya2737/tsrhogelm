import importlib
class Classify():

    def train(self,feature,nnc1,nnc2):
        train_mat   = self.runAdapter('elm','train',feature,nnc1,nnc2)
        return train_mat

    def test(self,feature):
        test_mat    = self.runAdapter('elm','test',feature,0,0)
        return test_mat
        ''''''
    def predict(self,feature):
        ''''''
    def runAdapter(self,method,type,feature,nnc1,nnc2):
        modulename = method
        module = importlib.import_module('.'+modulename,package='classifier')
        adapter = getattr(module, method)
        instance = adapter()
        try:
            return eval('instance.'+type+'(feature,nnc1,nnc2)')
        except:
            return eval('instance.'+type+'(feature)')

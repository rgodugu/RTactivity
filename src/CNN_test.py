import pickle
path = '/home/ravi/python/ML/data/train_data/CNN.pk1'
inp = open(path , 'rb')
parameters , accu = pickle.load(inp)
inp.close()

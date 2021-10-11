import torchtext

class BatchWrapper:
    
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

      def __iter__(self):
            for batch in self.dl:
                x = getattr(batch, self.x_var)
                y = batch.label
                yield (x, y)

      def __len__(self):
            return len(self.dl)

class CSVProcessor:
    
    def __init__(self, gpu, path, train, dev, max_size, min_freq, batch_size, split_sym, test=False):
        print('\n\nProcessing data...\n')
        self.device = 'cuda:0' if gpu == True else 'cpu'
        self.TEXT = torchtext.data.Field(lower=True, tokenize=lambda x: x.split(split_sym), use_vocab=True, batch_first=True)
        self.LABEL = torchtext.data.Field(sequential=False, is_target=True, batch_first=True)
        fields = [('text', self.TEXT), ('label', self.LABEL)]
        self.train, self.dev = torchtext.data.TabularDataset.splits(path=path,
                                                               train=train,
                                                               validation=dev,
                                                               fields=fields,
                                                               format='csv',
                                                               skip_header=True)
        self.TEXT.build_vocab(self.train, max_size=max_size, min_freq=min_freq)
        self.LABEL.build_vocab(self.train, self.dev)
        self.targets = self.LABEL.vocab
        self.vocab = self.TEXT.vocab
        train_iter = torchtext.data.BucketIterator(self.train,
                                                   batch_size,
                                                   train=True,
                                                   shuffle=True,
                                                   sort_key=lambda x: len(x.text),
                                                   sort_within_batch=True,
                                                   device=self.device)
        dev_iter = torchtext.data.BucketIterator(self.dev, batch_size, sort_key=lambda x: len(x.text), device=self.device)
        self.train_iter = BatchWrapper(train_iter, "text", "label")
        self.dev_iter = BatchWrapper(dev_iter, "text", "label")
        print('\ndone\n\n')

if __name__ == '__main__':
    
    gpu=False
    path = './data/'
    train = 'train.csv'
    dev = 'dev.csv'
    data = CSVProcessor(gpu=gpu,
                        path=path,
                        train=train,
                        dev=dev,
                        max_size=10000,
                        min_freq=2,
                        batch_size=2)
    print('test done')
    
        
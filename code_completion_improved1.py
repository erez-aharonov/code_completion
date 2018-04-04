import tflearn
import numpy
from code_completion_baseline import CodeCompletionBaseline


class CodeCompletionImproved1(CodeCompletionBaseline):
    def train(self, token_lists, model_file):
        (xs, ys) = self._prepare_data(token_lists)
        self._create_network()
        self.model.fit(xs, ys, n_epoch=3, batch_size=1024, show_metric=True)
        self.model.save(model_file)

    def _create_network(self):
        self.net = tflearn.input_data(shape=[None, len(self.string_to_number)])
        self.net = tflearn.fully_connected(self.net, 32)
        self.net = tflearn.fully_connected(self.net, 32)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)
    


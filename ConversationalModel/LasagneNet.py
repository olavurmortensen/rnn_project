from __future__ import absolute_import

from os.path import join
import os
from time import time
from datetime import datetime
from lasagne.layers import InputLayer

import lasagne
import numpy as np
from sklearn.cross_validation import train_test_split
import cPickle as pickle

class SaveParams(object):
    def __init__(self, name, save_dir, save_interval=10, file_name=None):
        self.name = name
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.file_name = file_name
        if not os.path.isdir(save_dir):
            raise Exception("params dir does not exist: %s" % self.save_dir)

    def __call__(self, nn, train_history):
        if len(train_history)%self.save_interval==0:
            if self.file_name is None:
                f_name = join(self.save_dir, "saved_params_%i" % len(train_history))
            else:
                f_name = join(self.save_dir, self.file_name)
            print "Saving params to %s" % f_name
            nn.save_params_to(f_name)

"""
_sldict returns slices of an array, or same slice from each element in a dictionary of arrays. E.g. if arr is
{X: X, X_mask: mask}, then {X: X[sl], X_mask: X[sl]} will be returned
"""
def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]


class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        bs = self.batch_size
        for i in range((self.n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = _sldict(self.X, sl)
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield (Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)


class LasagneNet:
    def __init__(self,
                 output_layer,
                 train_func,
                 test_func,
                 predict_func,
                 batch_iterator_train=BatchIterator(128),
                 batch_iterator_test=BatchIterator(128),
                 max_epochs=1000,
                 on_epoch_finished=None,
                 additional_params=None,
                 is_regression=False,
                 learning_rate_theano_shared=None,
                 **kwargs):
        self.max_epochs = max_epochs
        self.on_epoch_finished = on_epoch_finished or []
        self.additional_params = additional_params or {}

        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test

        self.output_layer = output_layer
        self.train_func = train_func
        self.test_func = test_func
        self.predict_func = predict_func
        self.is_regression = is_regression

        self.layers = {layer.name:layer for layer in lasagne.layers.get_all_layers(output_layer)}
        # information flows from parent to child...
        self.child_to_parent_map, self.parent_to_child_map, self.input_layers, self.output_layers = self.get_ancestor_maps(self.layers)
        self.output_layer = self.layers[self.output_layers[0]] # we assume that we only have one output layer.

        self.learning_rate = learning_rate_theano_shared

        all_trainable_parameters = lasagne.layers.get_all_params([self.output_layer], trainable=True)

        print "Trainable Model Parameters"
        print "-"*40
        total_num_param = 0
        for param in all_trainable_parameters:
            print param, param.get_value().shape
            total_num_param += np.prod(param.get_value().shape)
        print "-"*40
        print "Total number of trainable parameters: %i" % total_num_param


    def draw_network(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        import pylab

        G = nx.DiGraph()
        for parent in self.parent_to_child_map:
            for child in self.parent_to_child_map[parent]:
                G.add_edges_from([(parent, child)])




        edge_labels=dict([((u,v,),str(lasagne.layers.get_output_shape(self.layers[u]))) for u,v,d in G.edges(data=True)])

        pos=nx.graphviz_layout(G, root=self.input_layers)
        # pos=nx.fruchterman_reingold_layout(G)
        nx.draw_networkx_nodes(G,pos, dict([(u,u) for u,d in G.nodes(data=True)]),node_shape='s', node_size=3000)
        nx.draw_networkx_edges(G,pos,edge_labels,style='dotted')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw_networkx_labels(G, pos, dict([(u,u) for u,d in G.nodes(data=True)]))

        pylab.show()

    # information flows from parent to child (input_layer -> hidden -> output_layer.
    def get_ancestor_maps(self, layers):
        child_to_parent_map = {}
        parent_to_child_map = {}
        input_layers = []
        output_layers = []

        for layer_name in layers:
            if layer_name not in child_to_parent_map:
                child_to_parent_map[layer_name] = []
            if layer_name not in parent_to_child_map:
                parent_to_child_map[layer_name] = []
            if hasattr(layers[layer_name], 'input_layer'):
                parent_name = layers[layer_name].input_layer.name
                if parent_name not in parent_to_child_map:
                    parent_to_child_map[parent_name] = []
                child_to_parent_map[layer_name].append(parent_name)
                parent_to_child_map[parent_name ].append(layer_name)
            elif hasattr(layers[layer_name], 'input_layers'):
                for parent in layers[layer_name].input_layers:
                    if parent.name not in parent_to_child_map:
                        parent_to_child_map[parent.name] = []
                    child_to_parent_map[layer_name].append(parent.name)
                    parent_to_child_map[parent.name].append(layer_name)


        # an output layer is a layer with no children
        for name in parent_to_child_map:
            if len(parent_to_child_map[name]) == 0:
                output_layers.append(name)
        assert len(output_layers) == 1

        # an input layer is a layer with no parents
        for name in child_to_parent_map:
            if len(child_to_parent_map[name]) == 0:
                input_layers.append(name)
                assert isinstance(layers[name], InputLayer)

        return child_to_parent_map, parent_to_child_map, input_layers, output_layers


    """
    :return if X is a dict, it returns a train_dict, test_dict else (train_data_X, test_data_X)
    """
    def split_data(self, X, y=None, test_fraction=0.1):
        if isinstance(X, dict):
            dict_train = {}
            dict_test = {}
            N = X.values()[0].shape[0]
            test_idx = np.random.randint(0, N, int(test_fraction*N))
            train_idx = list(set(range(0,N)).difference(test_idx))
            for key in X:
                test_data = X[key][test_idx,::]
                train_data = X[key][train_idx,::]
                # train_data, test_data = train_test_split(X[key],test_size=0.1,random_state=0)
                dict_train[key] = train_data
                dict_test[key] = test_data
            X_train, X_test = dict_train, dict_test
        else:
            N = X.shape[0]
            test_idx = np.random.randint(0, N, int(test_fraction*N))
            train_idx = set(range(0,N)).difference(test_idx)
            X_test = X[test_idx,::]
            X_train = X[train_idx,::]
            # X_train, X_test = train_test_split(X,test_size=0.1,random_state=0), train_test_split(y,test_size=0.1,random_state=0)
        if y is not None:
            y_train = y[train_idx]
            y_test = y[test_idx]
            return X_train, X_test,y_train, y_test
        else:
            return X_train, X_test,None


    def fit(self, X, y=None):
        print "Fitting %i samples to model..." % y.shape[0]
        self.train_loop(X, y)
        return self



    def predict(self, X):
        batch_iterator = BatchIterator(1024*32)
        predictions = []
        for Xb, yb in batch_iterator(X):
            predictions += list(self.predict_func(**Xb)[0].reshape((-1,1)))
        return np.asarray(predictions)


    # def predict(self, X, batch_size):
    #     if isinstance(X, dict):
    #         predictions = []
    #         num_samples = X[X.keys()[0]].shape[0]
    #         batch_indices = range(0, num_samples, batch_size)
    #         for idx in batch_indices:
    #             dict_train = {}
    #             for key in X:
    #                 X_batch = X[key][slice(idx, idx+batch_size),::]
    #                 dict_train[key] = X_batch
    #             predictions += list(self.predict_func(**dict_train))
    #     else:
    #         raise NotImplementedError('Predict must be called with a dict')
    #     return np.asarray(predictions)



    def train_loop(self, X, y):
        X_train, X_test, y_train, y_test = self.split_data(X,y)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        epoch = 0
        best_valid_loss = np.inf
        best_train_loss = np.inf

        print "starting sampling..."
        self.train_history_ = []
        t0 = time()
        samples_processed = 0
        while epoch < self.max_epochs:
            train_costs, train_accuracies  = [], []
            valid_costs, valid_accuracies = [], []
            batch_counter = 0

            # start_idx = samples_processed % X_train.shape[0]
            # idx = slice(start_idx, (start_idx+self.BATCH_SIZE))
            # inputs, input_masks, targets = X_train[idx], masks_train[idx], y_train[idx]
            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                # batch_cost, batch_acc = self.train_func(Xb['X'], yb.reshape((1,-1)), Xb['X_mask'])
                batch_counter += 1
                if batch_counter % 200==0:
                    print ("Finished batch number %i" % batch_counter)
                Xb['y'] = yb.reshape((1,-1))
                Xb['y'] = yb

                batch_cost = self.train_func(**Xb)
                if self.is_regression:
                    batch_acc = -1
                else:
                    batch_cost, batch_acc = batch_cost
                if best_train_loss > batch_cost:
                    best_train_loss = batch_cost
                train_costs.append(batch_cost)
                train_accuracies.append(batch_acc)
                samples_processed += self.batch_iterator_train.batch_size

            for Xb, yb in self.batch_iterator_test(X_test, y_test):
                # val_cost, val_acc = self.test_func(Xb['X'], yb.reshape((1,-1)), Xb['X_mask'])
                Xb['y'] = yb
                val_cost = self.test_func(**Xb)
                valid_costs.append(val_cost)
                if self.is_regression:
                    val_acc = -1
                else:
                    val_cost, val_acc = val_cost
                valid_accuracies.append(val_acc)
                if best_valid_loss > val_cost:
                    best_valid_loss = val_cost

            samples_processed += self.batch_iterator_train.batch_size

            epoch += 1

            info = {
                'epoch': epoch,
                'train_loss_mean': np.mean(train_costs),
                'train_accuracy_mean': np.mean(train_accuracies),
                'valid_loss_mean': np.mean(valid_costs),
                'valid_accuracy_mean': np.mean(valid_accuracies),
                'valid_loss_best': best_valid_loss,
                'train_loss_best': best_train_loss,
                'dur': time() - t0,
                }
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

            remaining_epocs = (self.max_epochs-len(self.train_history_))
            time_pr_epoch = info['dur']/len(self.train_history_)
            print "Epoch number:%i" % epoch
            print "Time: " + str(datetime.now())
            print "Traing time until now: %1.1f sec." % info['dur']
            # print "Projected finish in : %1.4f min." % (remaining_epocs*time_pr_epoch/60)
            print "Average epoch time : %1.1f min." % (time_pr_epoch/60)
            print "Samples processed:%i" % samples_processed
            print "Validation cost: %1.4f" %  info['valid_loss_mean']
            print "Train cost: %1.4f" % info['train_loss_mean']

            if info['valid_accuracy_mean'] >= 0:
                print "Validation accuracy: %1.4f" %  info['valid_accuracy_mean']
                print "Train accuracy: %1.4f" % info['train_accuracy_mean']
            print "-"*40
            print


    def save_params_to(self, layer_weights_file, additional_params_file=None):
        params = lasagne.layers.get_all_param_values([self.output_layer], trainable=True)
        with open(layer_weights_file, 'wb') as f:
            pickle.dump(params, f)
        if additional_params_file != None:
            with open(additional_params_file, 'wb') as f:
                pickle.dump(self.additional_params, f)

    def load_weights_from(self, layer_weights_file, additional_params_file=None):
        with open(layer_weights_file, 'rb') as f:
            params = pickle.load(f)
        lasagne.layers.set_all_param_values(self.output_layer, params, trainable=True)
        if additional_params_file != None:
            with open(additional_params_file, 'rb') as f:
                self.additional_params = pickle.load(f)


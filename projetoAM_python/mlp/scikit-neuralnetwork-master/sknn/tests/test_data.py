import random
import unittest
from nose.tools import (assert_greater, assert_less, assert_raises, assert_equals, assert_true)

import logging

import numpy
from sknn.mlp import Regressor as MLPR, Classifier as MLPC
from sknn.mlp import Layer as L, Convolution as C


class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        self.called = 0
        self.value = 1.0

        self.nn = MLPR(
                    layers=[L("Linear")],
                    n_iter=1,
                    batch_size=1,
                    callback={'on_batch_start': self._mutate_fn})

    def _mutate_fn(self, Xb, **_):
        self.called += 1
        Xb[Xb == 0.0] = self.value

    def test_TestCalledOK(self):
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        self.nn._fit(a_in, a_out)
        assert_equals(a_in.shape[0], self.called)

    def test_DataIsUsed(self):
        self.value = float("nan")
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        assert_raises(RuntimeError, self.nn._fit, a_in, a_out)


class TestNetworkParameters(unittest.TestCase):
    
    def test_GetLayerParams(self):
        nn = MLPR(layers=[L("Linear")], n_iter=1)
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        p = nn.get_parameters()
        assert_equals(type(p), list)
        assert_true(isinstance(p[0], tuple))
        
        assert_equals(p[0].layer, 'output')
        assert_equals(p[0].weights.shape, (16, 4))
        assert_equals(p[0].biases.shape, (4,))

    def test_SetLayerParamsList(self):
        nn = MLPR(layers=[L("Linear")])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (16,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters([(weights, biases)])
        
        p = nn.get_parameters()
        assert_true((p[0].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[0].biases.astype('float32') == biases.astype('float32')).all())

    def test_LayerParamsSkipOneWithNone(self):
        nn = MLPR(layers=[L("Sigmoid", units=32), L("Linear", name='abcd')])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (32,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters([None, (weights, biases)])
        
        p = nn.get_parameters()
        assert_true((p[1].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[1].biases.astype('float32') == biases.astype('float32')).all())

    def test_SetLayerParamsDict(self):
        nn = MLPR(layers=[L("Sigmoid", units=32), L("Linear", name='abcd')])
        a_in, a_out = numpy.zeros((8,16)), numpy.zeros((8,4))
        nn._initialize(a_in, a_out)
        
        weights = numpy.random.uniform(-1.0, +1.0, (32,4))
        biases = numpy.random.uniform(-1.0, +1.0, (4,))
        nn.set_parameters({'abcd': (weights, biases)})
        
        p = nn.get_parameters()
        assert_true((p[1].weights.astype('float32') == weights.astype('float32')).all())
        assert_true((p[1].biases.astype('float32') == biases.astype('float32')).all())


class TestMaskedDataRegression(unittest.TestCase):

    def check(self, a_in, a_out, a_mask):
        nn = MLPR(layers=[L("Linear")], learning_rule='adam', n_iter=50)
        nn.fit(a_in, a_out, a_mask)
        v_out = nn.predict(a_in)

        # Make sure the examples weighted 1.0 have low error, 0.0 high error.
        print(abs(a_out - v_out).T * a_mask)
        assert_true((abs(a_out - v_out).T * a_mask < 1E-1).all())
        assert_true((abs(a_out - v_out).T * (1.0 - a_mask) > 2.5E-1).any())

    def test_SingleOutputOne(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_mask = (0.0 + a_out).flatten()
        
        self.check(a_in, a_out, a_mask)

    def test_SingleOutputZero(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_mask = (1.0 - a_out).flatten()

        self.check(a_in, a_out, a_mask)

    def test_SingleOutputNegative(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,1)).astype(numpy.float32)
        a_mask = (0.0 + a_out).flatten()
        a_out = -1.0 * 2.0 + a_out
        
        self.check(a_in, a_out, a_mask)
        
    def test_MultipleOutputRandom(self):
        a_in = numpy.random.uniform(-1.0, +1.0, (8,16))
        a_out = numpy.random.randint(2, size=(8,4)).astype(numpy.float32)
        a_mask = numpy.random.randint(2, size=(8,)).astype(numpy.float32)

        self.check(a_in, a_out, a_mask)


class TestMaskedDataClassification(unittest.TestCase):

    def check(self, a_in, a_out, a_mask, act='Softmax', n_iter=100):
        nn = MLPC(layers=[L(act)], learning_rule='rmsprop', n_iter=n_iter)
        nn.fit(a_in, a_out, a_mask)
        return nn.predict_proba(a_in)

    def test_TwoLabelsOne(self):
        # Only one sample has the value 1 with weight 1.0, but all 0s are weighted 0.0.
        a_in = numpy.random.uniform(-1.0, +1.0, (16,4))
        a_out = numpy.zeros((16,1), dtype=numpy.int32)
        a_out[0] = 1
        a_mask = (0.0 + a_out).flatten()
        
        a_test = self.check(a_in, a_out, a_mask).mean(axis=0)
        assert_greater(a_test[1], a_test[0] * 1.25)

    def test_TwoLabelsZero(self):
        # Only one sample has the value 0 with weight 1.0, but all 1s are weighted 0.0. 
        a_in = numpy.random.uniform(-1.0, +1.0, (16,4))
        a_out = numpy.ones((16,1), dtype=numpy.int32)
        a_out[-1] = 0
        a_mask = (1.0 - a_out).flatten()
        
        a_test = self.check(a_in, a_out, a_mask).mean(axis=0)
        assert_greater(a_test[0], a_test[1] * 1.25)

    def test_FourLabels(self):
        # Only multi-label sample has weight 1.0, the others have weight 0.0. Check probabilities!
        chosen = random.randint(0,15)
        a_in = numpy.random.uniform(-1.0, +1.0, (16,4))
        a_out = numpy.random.randint(2, size=(16,4))
        a_mask = numpy.zeros((16,), dtype=numpy.int32)
        a_mask[chosen] = 1.0

        a_test = self.check(a_in, a_out, a_mask, act="Sigmoid", n_iter=250).mean(axis=0)
        for i in range(a_out.shape[1]):
            compare = assert_greater if a_out[chosen][i]==0 else assert_less
            compare(a_test[i*2], a_test[i*2+1])

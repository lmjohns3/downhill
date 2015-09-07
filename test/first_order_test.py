import numpy as np

import util


class TestSGD:
    def test_rosen(self):
        util.assert_progress(
            *util.build_rosen('sgd'),
            monitor_gradients=True)

    def test_factor(self):
        util.assert_progress(
            *util.build_factor('sgd'),
            max_gradient_elem=1,
            nesterov=False)

    def test_factor_nesterov(self):
        util.assert_progress(
            *util.build_factor('sgd'),
            max_gradient_norm=1)

    def test_default_params(self):
        opt, data = util.build_rosen('sgd')
        for _ in opt.iterate(data):
            assert opt.nesterov is False
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.momentum, 0)
            assert np.allclose(opt.patience, 5)
            assert np.allclose(opt.min_improvement, 0)
            assert np.allclose(opt.max_gradient_norm, 0)
            assert np.allclose(opt.max_gradient_elem, 0)
            break

    def test_params(self):
        opt, data = util.build_rosen('sgd')
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             momentum=10,
                             patience=20,
                             min_improvement=0.1,
                             max_gradient_elem=4,
                             max_gradient_norm=5,
                             nesterov=True):
            assert opt.nesterov is True
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.momentum, 10)
            assert np.allclose(opt.patience, 20)
            assert np.allclose(opt.min_improvement, 0.1)
            assert np.allclose(opt.max_gradient_norm, 5)
            assert np.allclose(opt.max_gradient_elem, 4)
            break


class TestNAG:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('nag'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('nag'), max_gradient_elem=1)

    def test_default_params(self):
        opt, data = util.build_rosen('nag')
        for _ in opt.iterate(data):
            assert opt.nesterov is True
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.momentum, 0)
            assert np.allclose(opt.patience, 5)
            assert np.allclose(opt.min_improvement, 0)
            assert np.allclose(opt.max_gradient_norm, 0)
            assert np.allclose(opt.max_gradient_elem, 0)
            break

    def test_params(self):
        opt, data = util.build_rosen('nag')
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             momentum=10,
                             patience=20,
                             min_improvement=0.1,
                             max_gradient_elem=4,
                             max_gradient_norm=5,
                             nesterov=False):
            assert opt.nesterov is True  # nesterov always True for NAG
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.momentum, 10)
            assert np.allclose(opt.patience, 20)
            assert np.allclose(opt.min_improvement, 0.1)
            assert np.allclose(opt.max_gradient_norm, 5)
            assert np.allclose(opt.max_gradient_elem, 4)
            break

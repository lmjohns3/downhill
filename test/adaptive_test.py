import numpy as np

import util


class TestESGD:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('esgd'), learning_rate=1e-6)

    def test_factor(self):
        util.assert_progress(*util.build_factor('esgd'), learning_rate=1e-6)

    def test_default_params(self):
        opt, data = util.build_rosen('esgd')
        assert opt.hv_method == 'rop'
        for _ in opt.iterate(data):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 14))
            assert np.allclose(opt.epsilon.eval(), 1e-8)
            break

    def test_params(self):
        opt, data = util.build_rosen('esgd')
        opt.hv_method = 'lop'  # TODO(leif): incorporate into downhill.build()?
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             rms_halflife=10,
                             rms_regularizer=20):
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 10))
            assert np.allclose(opt.epsilon.eval(), 20)
            break


class TestRProp:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('rprop'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('rprop'))

    def test_default_params(self):
        opt, data = util.build_rosen('rprop')
        for _ in opt.iterate(data):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.step_increase.eval(), 1.01)
            assert np.allclose(opt.step_decrease.eval(), 0.99)
            assert np.allclose(opt.min_step.eval(), 0)
            assert np.allclose(opt.max_step.eval(), 100)
            break

    def test_params(self):
        opt, data = util.build_rosen('rprop')
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             rprop_increase=22,
                             rprop_decrease=101,
                             rprop_min_step=50,
                             rprop_max_step=-10):
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.step_increase.eval(), 22)
            assert np.allclose(opt.step_decrease.eval(), 101)
            assert np.allclose(opt.min_step.eval(), 50)
            assert np.allclose(opt.max_step.eval(), -10)
            break


class TestADAGRAD:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('adagrad'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('adagrad'))

    def test_default_params(self):
        opt, data = util.build_rosen('adagrad')
        for _ in opt.iterate(data):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.epsilon.eval(), 1e-8)
            break

    def test_params(self):
        opt, data = util.build_rosen('adagrad')
        for _ in opt.iterate(data, rms_regularizer=0.1):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.epsilon.eval(), 0.1)
            break


class TestRMSProp:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('rmsprop'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('rmsprop'))

    def test_default_params(self):
        opt, data = util.build_rosen('rmsprop')
        for _ in opt.iterate(data):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 14))
            assert np.allclose(opt.epsilon.eval(), 1e-8)
            break

    def test_params(self):
        opt, data = util.build_rosen('rmsprop')
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             rms_halflife=10,
                             rms_regularizer=20):
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 10))
            assert np.allclose(opt.epsilon.eval(), 20)
            break


class TestADADELTA:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('adadelta'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('adadelta'))

    def test_default_params(self):
        opt, data = util.build_rosen('adadelta')
        for _ in opt.iterate(data):
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 14))
            assert np.allclose(opt.epsilon.eval(), 1e-8)
            break

    def test_params(self):
        opt, data = util.build_rosen('adadelta')
        for _ in opt.iterate(data,
                             rms_halflife=10,
                             rms_regularizer=20):
            assert np.allclose(opt.ewma.eval(), np.exp(-np.log(2) / 10))
            assert np.allclose(opt.epsilon.eval(), 20)
            break


class TestAdam:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('adam'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('adam'))

    def test_default_params(self):
        opt, data = util.build_rosen('adam')
        for _ in opt.iterate(data):
            assert np.allclose(opt.learning_rate.eval(), 1e-4)
            assert np.allclose(opt.beta1_decay.eval(), 1 - 1e-6)
            assert np.allclose(opt.beta1.eval(), np.exp(-np.log(2) / 7))
            assert np.allclose(opt.beta2.eval(), np.exp(-np.log(2) / 69))
            assert np.allclose(opt.epsilon.eval(), 1e-8)
            break

    def test_params(self):
        opt, data = util.build_rosen('adam')
        for _ in opt.iterate(data,
                             learning_rate=0.3,
                             beta1_decay=0.5,
                             beta1_halflife=10,
                             beta2_halflife=20,
                             rms_regularizer=11):
            assert np.allclose(opt.learning_rate.eval(), 0.3)
            assert np.allclose(opt.beta1_decay.eval(), 0.5)
            assert np.allclose(opt.beta1.eval(), np.exp(-np.log(2) / 10))
            assert np.allclose(opt.beta2.eval(), np.exp(-np.log(2) / 20))
            assert np.allclose(opt.epsilon.eval(), 11)
            break

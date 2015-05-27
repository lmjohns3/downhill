import downhill

import util


class TestBuild:
    def test_sgd(self):
        assert isinstance(util.build_rosen('sgd'), downhill.SGD)
        assert isinstance(util.build_factor('sgd'), downhill.SGD)

    def test_nag(self):
        assert isinstance(util.build_rosen('nag'), downhill.NAG)

    def test_rmsprop(self):
        assert isinstance(util.build_rosen('RmsProp'), downhill.RMSProp)

    def test_adadelta(self):
        assert isinstance(util.build_rosen('ADADELTA'), downhill.ADADELTA)

    def test_esgd(self):
        assert isinstance(util.build_rosen('EsGd'), downhill.ESGD)


class Tester(downhill.Optimizer):
    def _get_updates_for(self, param, grad):
        yield (param, param + 1.1)


class TestOptimizer:
    def test_tester(self):
        opt = util.build_rosen('tester')
        assert isinstance(opt, Tester)

        # run the optimizer for three iterations. check that the x and y values
        # (being monitored) increase at each iteration.
        for i, (tm, vm) in enumerate(opt.iteropt([[]])):
            assert tm['x'] >= vm['x']
            assert tm['y'] >= vm['y']
            if i == 2:
                break

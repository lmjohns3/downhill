import downhill

import util


class TestBuild:
    def test_sgd(self):
        assert isinstance(util.build_rosen('sgd')[0], downhill.SGD)
        assert isinstance(util.build_factor('sgd')[0], downhill.SGD)

    def test_nag(self):
        assert isinstance(util.build_rosen('nag')[0], downhill.NAG)

    def test_rprop(self):
        assert isinstance(util.build_rosen('RProp')[0], downhill.RProp)

    def test_rmsprop(self):
        assert isinstance(util.build_rosen('RmsProp')[0], downhill.RMSProp)

    def test_adadelta(self):
        assert isinstance(util.build_rosen('ADADELTA')[0], downhill.ADADELTA)

    def test_esgd(self):
        assert isinstance(util.build_rosen('EsGd')[0], downhill.ESGD)

    def test_adam(self):
        assert isinstance(util.build_rosen('Adam')[0], downhill.Adam)


class Tester(downhill.Optimizer):
    def _get_updates_for(self, param, grad):
        yield (param, param + 1.1)


class TestOptimizer:
    def test_rosen(self):
        opt, train = util.build_rosen('tester')
        assert isinstance(opt, Tester)

        # run the optimizer for three iterations. check that the x and y values
        # (being monitored) increase at each iteration.
        for i, (tm, vm) in enumerate(opt.iteropt(train)):
            assert tm['x'] >= vm['x']
            assert tm['y'] >= vm['y']
            if i == 2:
                break

    def test_factor(self):
        opt, train = util.build_factor('tester')
        assert isinstance(opt, Tester)

        # run the optimizer for two iterations. check that the u and v values
        # (being monitored) are reasonable at the start.
        for i, (tm, vm) in enumerate(opt.iteropt(train)):
            assert abs(vm['u<1'] - 0.001) < 1e-5
            assert vm['u<-1'] == 0
            assert vm['v<1'] == 1
            assert vm['v<-1'] == 0
            if i == 2:
                break

    def test_gradient_clip(self):
        opt, data = util.build_rosen('tester')
        for _ in opt.iteropt(data, gradient_clip=1):
            assert opt.max_gradient_elem == 1
            break
        for _ in opt.iteropt(data, max_gradient_clip=2):
            assert opt.max_gradient_elem == 2
            break
        for _ in opt.iteropt(data, max_gradient_elem=3):
            assert opt.max_gradient_elem == 3
            break

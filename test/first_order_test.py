import util


class TestSGD:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('sgd'))

    def test_factor(self):
        util.assert_progress(
            *util.build_factor('sgd'),
            max_gradient_clip=1,
            nesterov=False)

    def test_factor_nesterov(self):
        util.assert_progress(
            *util.build_factor('sgd'),
            max_gradient_clip=1)


class TestNAG:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('nag'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('nag'), max_gradient_clip=1)

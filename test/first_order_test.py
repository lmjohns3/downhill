import util


class TestSGD:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('sgd'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('sgd'))


class TestNAG:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('nag'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('nag'))

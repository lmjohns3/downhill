import util


class TestSGD:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('sgd'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('sgd'), learning_rate=0.00001)


class TestNAG:
    def test_rosen(self):
        util.assert_progress(*util.build_rosen('nag'))

    def test_factor(self):
        util.assert_progress(*util.build_factor('nag'), learning_rate=0.00001)

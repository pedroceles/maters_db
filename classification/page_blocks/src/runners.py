import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA


class PageBlocksImputationRunner(ImputationComparisonRunner):
    base_name = "PageBlocks"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(PageBlocksImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(PageBlocksImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class PageBlocksMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(PageBlocksMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class PageBlocksMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = PageBlocksImputationRunner


class PageBlocksMultipleEstimators(MultipleEstimatorsStudy):
    runner = PageBlocksImputationRunner


class PageBlocksMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = PageBlocksMissingComparisonRunner

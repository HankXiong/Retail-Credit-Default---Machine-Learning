# -*- coding: utf-8 -*-

import abc


class BaseDecompositionClass:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def decompose(self, dataframe):
        """
        :param dataframe: a pandas dataframe that includes feature space to be decomposed
        :return:
        """

        pass
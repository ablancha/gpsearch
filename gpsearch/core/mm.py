import numpy as np
from .utils import custom_KDE, compute_mean, compute_mean_jac
from .minimizers import funmin


class Metric(object):

    need_recommendation = False

    def __init__(self, tmap, inputs, kwargs={}):
        self.tmap = tmap
        self.inputs = inputs
        self.kwargs = kwargs

    def evaluate_list(self, m_list, rec_list=None):

        if rec_list is None:
            res = [ self.evaluate(model, **self.kwargs)
                    for (model, x_rec) in m_list) ]
        else:
            res = [ self.evaluate(model, x_rec, **self.kwargs)
                    for (model, x_rec) in zip(m_list, rec_list) ]

        return res

    def evaluate(self, model):
        raise NotImplementedError


class RegretBlackBox(MetricWithRecommendation):

    need_recommendation = True

    def evaluate(self, model, x_rec, true_ymin=0):
        res = np.abs(self.tmap.evaluate(x_rec) - true_ymin)
        return res


class RegretModel(Metric):

    need_recommendation = True

    def evaluate(self, model, x_rec, true_ymin=0):
        res = np.abs(compute_mean(x_rec, model) - true_ymin)
        return res


class RegretObservation(Metric):

    need_recommendation = False

    def evaluate(self, model, x_rec, true_ymin=0):
        res = np.abs(model.Y.min() - true_ymin)
        return res


class DistMinModel(Metric):

    need_recommendation = True

    def evaluate(self, model, x_rec, true_xmin=None):
        res = min([ np.linalg.norm(x_rec - true_x) 
                    for true_x in true_xmin ])
    return res


class DistMinObservation(Metric):

    need_recommendation = False

    def evaluate(self, model, x_rec, true_xmin=None):
        x_rec = model.X[np.argmin(model.Y)]
        res = min([ np.linalg.norm(x_rec - true_x) 
                    for true_x in true_xmin ])
    return res


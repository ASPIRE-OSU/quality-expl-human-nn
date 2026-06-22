#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Feature importance explanations.
"""
import numpy as np
from ..base import ExplanationBase, DashFigure
import pandas as pd

class FeatureImportance(ExplanationBase):
    """
    The class for feature importance explanations. It uses a list to store
    the feature importance explanations of the input instances. Each item in the list
    is a dict with the following format `{"instance": the input instance, "features": a list of feature names,
    "values": a list of feature values, "scores": a list of feature importance scores}`.
    If the task is `classification`, the dict has an additional entry `{"target_label":
    the predicted label of the input instance}`.
    """

    def __init__(self, mode, explanations=None):
        """
        :param mode: The task type, e.g., `classification` or `regression`.
        :param explanations: The explanation results for initializing ``FeatureImportance``,
            which is optional.
        """
        super().__init__()
        self.mode = mode
        self.explanations = [] if explanations is None else explanations

    def __repr__(self):
        return repr(self.explanations)

    def __getitem__(self, i: int):
        assert i < len(self.explanations)
        return FeatureImportance(mode=self.mode, explanations=[self.explanations[i]])

    def add(self, instance, target_label, feature_names, feature_values, importance_scores, sort=False, **kwargs):
        """
        Adds the generated explanation corresponding to one instance.

        :param instance: The instance to explain.
        :param target_label: The label to explain, which is ignored for regression.
        :param feature_names: The list of the feature column names.
        :param feature_values: The list of the feature values.
        :param importance_scores: The list of the feature importance scores.
        :param sort: `True` if the features are sorted based on the importance scores.
        """
        scores = list(zip(feature_names, feature_values, importance_scores))
        if sort:
            scores = sorted(scores, key=lambda x: abs(x[-1]), reverse=True)
        e = {
            "instance": instance,
            "features": [s[0] for s in scores],
            "values": [s[1] for s in scores],
            "scores": [s[2] for s in scores],
        }
        e.update(kwargs)
        if self.mode == "classification":
            e["target_label"] = target_label
        self.explanations.append(e)

    def get_explanations(self, index=None):
        """
        Gets the generated explanations.

        :param index: The index of an explanation result stored in ``FeatureImportance``.
            When ``index`` is None, the function returns a list of all the explanations.
        :return: The explanation for one specific instance (a dict)
            or the explanations for all the instances (a list of dicts).
            Each dict has the following format: `{"instance": the input instance,
            "features": a list of feature names, "values": a list of feature values,
            "scores": a list of feature importance scores}`. If the task is `classification`,
            the dict has an additional entry `{"target_label": the predicted label
            of the input instance}`.
        :rtype: Union[Dict, List]
        """
        return self.explanations if index is None else self.explanations[index]

    def plot(self, output_path='', save_expl=True, index=None, class_names=None, num_features=None, max_num_subplots=None, files=None, suppress_image=True, **kwargs):
        """
        Plots feature importance scores.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            e.g., it will plot the first explanation result when ``index = 0``.
            When ``index`` is None, it shows a figure with ``max_num_subplots`` subplots
            where each subplot plots the feature importance scores for one instance.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        :param max_num_subplots: The maximum number of subplots in the figure.
        :return: A matplotlib figure plotting feature importance scores.
        """
        import matplotlib.pyplot as plt

        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if max_num_subplots is not None:
            indices = indices[:max_num_subplots]

        num_rows = int(np.round(np.sqrt(len(indices))))
        num_cols = int(np.ceil(len(indices) / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        print('Plotting explanations...')
        for i, index in enumerate(indices):
            exp = explanations[index]
            feat_scores = sorted(
                list(zip([f"{self._s(f)} = {self._s(v)}    "
                          for f, v in zip(exp["features"], exp["values"])], exp["scores"])),
                key=lambda x: abs(x[1]),
            )
            all_scores = [s for f,s in feat_scores]
            num_nonzero = np.count_nonzero(all_scores)
            
            # default to graphing 20 features
            if num_features is None:
                num_features = 20
            print('Graphing top {} features of total {}'.format(num_features, num_nonzero))
            feat_scores_all = feat_scores   # we want to save all of the features in the csv below
            feat_scores = feat_scores[-num_features:]
            
            # get feature names
            # f, s in feat_scores: f format is 'freqbin_### = [...]'
            fnames = [f.split(' ')[0] for f,s in feat_scores if s != 0.0]    
            fnames_all = [f.split(' ')[0] for f,s in feat_scores_all]
            
            # Ignore those features with importance_score = 0
            scores = [s for f, s in feat_scores if s != 0.0]
            scores_all = [s for f, s in feat_scores_all]
            if save_expl:
                #self.save_feat_scores(scores, fnames, output_path.split('.')[0] + '_feat_scores.csv', num_feat=len(all_scores))    
                self.save_feat_scores(scores_all, fnames_all, output_path + '_{}_feat_scores.csv'.format(files[i]), num_feat=len(all_scores), filename=files[i])
            
            colors = ["red" if x > 0 else "blue" for x in scores]
            positions = np.arange(len(scores)) + 0.5

            row, col = divmod(i, num_cols)
            plt.sca(axes[row, col])
            plt.barh(positions, scores, align="center", color=colors)
            axes[row, col].yaxis.set_ticks_position("left")
            plt.yticks(positions, fnames, ha="right")
            
            for id, v in enumerate(scores):
                plt.text(v, id, '{0:.4f}'.format(v), ha='left')
            
            if self.mode == "classification":
                class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
                plt.title(f"Instance {index}: Class {class_name}")
            else:
                plt.title(f"Instance {index}")
            print(output_path + '_{}.png'.format(files[i]))
            
            if not suppress_image:
                plt.savefig(output_path + '_{}.png'.format(files[i]), bbox_inches='tight')
        return fig
    
    def save_feat_scores(self, scores, fnames, output_path, num_feat=None, filename=None, verbose=False):
        if num_feat is None:
            num_feat = len(fnames)
            
        feat_type = fnames[0].split('_')[0]
            
        idx = [int(f.split('_')[-1]) for f in fnames]
        
        df = pd.DataFrame(columns=['file'] + fnames)
        vals = [0] * len(fnames)
        for i in range(len(idx)):
            if verbose:
                print('vals: ', vals)
                print('len vals: ', len(vals))
                print('idx[i]: ', idx[i])
                print('len idx: ', len(idx))
                print('i: ', i)
                print('scores: ', scores)
                print('len scores: ', len(scores))
            vals[idx[i]] = scores[i]
            
        if filename is None:
            df.loc[0] = vals
        else:
            df.loc[0] = [filename.split('.h5')[0]] + vals
        
        fcols = [col for col in df.columns if col != 'file']
        fcols_sorted = sorted(fcols, key=lambda x: int(x.split('_')[1]))
        new_order = ['file'] + fcols_sorted
        df = df[new_order]
        print(output_path)
        df.to_csv(output_path, index=False)

    # assumes all explanations in self are for the same file
    def average_explanation(self, fnames, filename, index=None, output_path='', max_num_subplots=4, num_features=None, save_expl=True):
        explanations = self.get_explanations(index)
        explanations = (
            {index: explanations} if isinstance(explanations, dict) else {i: e for i, e in enumerate(explanations)}
        )
        indices = sorted(explanations.keys())
        if max_num_subplots is not None:
            indices = indices[:max_num_subplots]
            
        all_scores = {i: ([], []) for i in fnames}

        for i, index in enumerate(indices):
            exp = explanations[index]
            
            feat_scores = sorted(
                list(zip(exp["features"], exp["values"], exp["scores"])),
                key=lambda x: abs(x[2])
            )
            
            for f in feat_scores:
                all_scores[f[0]][0].append(f[1])
                all_scores[f[0]][1].append(f[2])
                #all_scores = [s for f,s in feat_scores]
        
        averaged = {}
        for f, (v, s) in all_scores.items():
            averaged[f] = (np.mean(v), np.mean(s))
        #np.avg(all_scores[i]) for i in all_scores.keys()}
        
        # {"instance": the input instance, "features": a list of feature names,
        #    "values": a list of feature values, "scores": a list of feature importance scores}`
        average_expl = {"instance": filename,
                        "features": averaged.keys(),
                        "values": [averaged[k][0] for k in averaged.keys()],
                        "scores": [averaged[k][1] for k in averaged.keys()]}
        
        # plot!
        import matplotlib.pyplot as plt
        
        num_rows = 1 #int(np.round(np.sqrt(len(indices))))
        num_cols = 1 #int(np.ceil(len(indices) / num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)
        
        feat_scores = sorted(
            list(zip([f"{self._s(f)} = {self._s(v)}    " for f, v in zip(average_expl["features"], average_expl["values"])], 
                        average_expl["scores"])),
            key=lambda x: abs(x[1]),
        )
         
        all_scores = [s for f,s in feat_scores]        
        num_nonzero = np.count_nonzero(all_scores)
        
        # default to graphing 20 features
        if num_features is None:
            num_features = 20
        print('Graphing top {} features of total {}'.format(num_features, num_nonzero))
       
        #feat_scores = feat[-num_features:]
        feat_scores_all = feat_scores   # we want to save all of the features in the csv below
        feat_scores = feat_scores[-num_features:]
            
        # get feature names
        # f, s in feat_scores: f format is 'freqbin_### = [...]'
        fnames = [f.split(' ')[0] for f,s in feat_scores if s != 0.0]    
        fnames_all = [f.split(' ')[0] for f,s in feat_scores_all]
        
        # Ignore those features with importance_score = 0
        scores = [s for f, s in feat_scores if s != 0.0]
        scores_all = [s for f, s in feat_scores_all]
            
        if save_expl:
            self.save_feat_scores(scores_all, fnames_all, output_path + '_{}_feat_scores.csv'.format(filename), num_feat=len(all_scores), filename=filename)
            
        colors = ["red" if x > 0 else "blue" for x in scores]
        positions = np.arange(len(scores)) + 0.5

        #row, col = divmod(0, num_cols)
        row, col = (0, 0)
        plt.sca(axes[row, col])
        plt.barh(positions, scores, align="center", color=colors)
        axes[row, col].yaxis.set_ticks_position("left")
        plt.yticks(positions, fnames, ha="right")
        
        for id, v in enumerate(scores):
            plt.text(v, id, '{0:.4f}'.format(v), ha='left')
            
        plt.title(f"Instance {filename}")
        print(output_path + '_{}.png'.format(filename))
        plt.savefig(output_path + '_{}.png'.format(filename), bbox_inches='tight')
        return fig

    def _plotly_figure(self, index, class_names=None, num_features=20, **kwargs):
        import plotly.express as px

        exp = self.explanations[index]
        if self.mode == "classification":
            class_name = exp["target_label"] if class_names is None else class_names[exp["target_label"]]
            title = f"Label: Class {class_name}"
        else:
            title = ""

        feat_scores = sorted(
            list(zip([f"{self._s(f)} = {self._s(v)}"
                      for f, v in zip(exp["features"], exp["values"])], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores if s != 0.0]
        scores = [s for f, s in feat_scores if s != 0.0]

        fig = px.bar(
            y=fnames,
            x=scores,
            orientation="h",
            color=[s > 0 for s in scores],
            labels={"color": "Positive", "x": "Importance scores", "y": "Features"},
            title=title,
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        return fig

    def plotly_plot(self, index=0, class_names=None, num_features=20, **kwargs):
        """
        Plots feature importance scores for one specific instance using Dash.

        :param index: The index of an explanation result stored in ``FeatureImportance``
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        :return: A plotly dash figure plotting feature importance scores.
        """
        assert index is not None, "`index` cannot be None for `plotly_plot`. " "Please specify the instance index."
        return DashFigure(self._plotly_figure(index, class_names=class_names, num_features=num_features, **kwargs))

    def ipython_plot(self, index=0, class_names=None, num_features=20, **kwargs):
        """
        Plots the feature importance scores in IPython.

        :param index: The index of an explanation result stored in ``FeatureImportance``,
            which cannot be None, e.g., it will plot the first explanation result
            when ``index = 0``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param num_features: The maximum number of features to plot.
        """
        import plotly

        assert index is not None, "`index` cannot be None for `ipython_plot`. " "Please specify the instance index."
        plotly.offline.iplot(self._plotly_figure(index, class_names=class_names, num_features=num_features, **kwargs))

    @classmethod
    def from_dict(cls, d):
        import pandas as pd
        explanations = []
        for e in d["explanations"]:
            e["instance"] = pd.DataFrame.from_dict(e["instance"])
            explanations.append(e)
        return FeatureImportance(mode=d["mode"], explanations=explanations)


class GlobalFeatureImportance(ExplanationBase):
    """
    The class for global feature importance scores. It uses a dict to store
    the feature importance scores with the following format `{"features": a list of feature names,
    "scores": a list of feature importance scores}`.
    """

    def __init__(self):
        super().__init__()
        self.explanations = {}

    def add(self, feature_names, importance_scores, sort=False, **kwargs):
        """
        Adds the generated feature importance scores.

        :param feature_names: The list of the feature column names.
        :param importance_scores: The list of the feature importance scores.
        :param sort: `True` if the features are sorted based on the importance scores.
        """
        scores = list(zip(feature_names, importance_scores))
        if sort:
            scores = sorted(scores, key=lambda x: abs(x[-1]), reverse=True)
        self.explanations = {
            "features": [s[0] for s in scores],
            "scores": [s[1] for s in scores],
        }

    def get_explanations(self):
        """
        Gets the generated explanations.

        :return: The feature importance scores.
            The returned dict has the following format: `{"features": a list of feature names,
            "scores": a list of feature importance scores}`.
        :rtype: Dict
        """
        return self.explanations

    def plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots feature importance scores.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        :return: A matplotlib figure plotting feature importance scores.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1)
        exp = self.get_explanations()
        feat_scores = sorted(
            list(zip([f"{self._s(f) if truncate_long_features else f}    " for f in exp["features"]], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores]
        scores = [s for f, s in feat_scores]
        colors = ["green" if x > 0 else "red" for x in scores]
        positions = np.arange(len(scores)) + 0.5

        plt.sca(axes)
        plt.barh(positions, scores, align="center", color=colors)
        axes.yaxis.set_ticks_position("right")
        plt.yticks(positions, fnames, ha="right")
        plt.title(f"Global Feature Importance")
        return fig

    def _plotly_figure(self, num_features=20, truncate_long_features=True, **kwargs):
        import plotly.express as px

        exp = self.explanations
        title = f"Global Feature Importance"
        feat_scores = sorted(
            list(zip([f"{self._s(f) if truncate_long_features else f}" for f in exp["features"]], exp["scores"])),
            key=lambda x: abs(x[1]),
        )
        if num_features is not None:
            feat_scores = feat_scores[-num_features:]
        fnames = [f for f, s in feat_scores]
        scores = [s for f, s in feat_scores]

        fig = px.bar(
            y=fnames,
            x=scores,
            orientation="h",
            labels={"x": "Importance scores", "y": "Features"},
            title=title,
            color_discrete_map={True: "#008B8B", False: "#DC143C"},
        )
        return fig

    def plotly_plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots feature importance scores for one specific instance using Dash.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        :return: A plotly dash figure plotting feature importance scores.
        """
        return DashFigure(self._plotly_figure(num_features=num_features,
                                              truncate_long_features=truncate_long_features,
                                              **kwargs))

    def ipython_plot(self, num_features=20, truncate_long_features=True, **kwargs):
        """
        Plots the feature importance scores in IPython.

        :param num_features: The maximum number of features to plot.
        :param truncate_long_features: Flag to truncate long feature names
        """
        import plotly
        plotly.offline.iplot(self._plotly_figure(num_features=num_features,
                                                 truncate_long_features=truncate_long_features,
                                                 **kwargs))

    @classmethod
    def from_dict(cls, d):
        exp = GlobalFeatureImportance()
        exp.explanations = d["explanations"]
        return exp

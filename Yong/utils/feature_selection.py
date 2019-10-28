"""
Maximum Relevance Minimum Redundancy Markov Blanket Method
Designed by Yong Zhuang
"""
import dcor
import pandas as pd
import numpy as np

#Maximum Relevance Minimum Redundancy Markov Blanket(MRMRMB)
class MRMRMB:
    # data and target should be dataframe
    def get_mb(self, data, target, prob=0.95):
        alpha = 1.0 - prob  # significance level 0.05 if prob = 0.95
        # implementation is here
        self.data = data.astype(float)
        self.col_names = data.columns
        # self.mb = [] #np.array([],dtype='int')
        self.temp_mb1 = pd.DataFrame(columns=["idx", "name", "p_value", "statistic"])
        self.temp_mb2 = pd.DataFrame(columns=["idx", "name", "p_value", "statistic"])
        self.mb = pd.DataFrame(columns=["idx", "name", "corr"])
        self.n_r, self.n_c = self.data.shape
        print("the original data space has total "+str(self.n_c)+" features")
        # Phase I: remove irrelevance
        print("start phase I: remove irrelevance")
        for i in range(self.n_c):
            with np.errstate(divide="ignore"):
                # The null hypothesis is that the two random vectors are independent.
                hypothesis_test = dcor.independence.distance_correlation_t_test(data.iloc[:, i].values, target.values)
                # p_value <= alpha Dependent (reject H0), otherwise Independent (fail to reject H0),
                if hypothesis_test.p_value <= alpha:
                    self.temp_mb1 = self.temp_mb1.append(
                        {
                            "idx": i,
                            "name": self.col_names[i],
                            "p_value": hypothesis_test.p_value,
                            "statistic": hypothesis_test.statistic,
                        },
                        ignore_index=True,
                    )
                else:
                    continue
        self.temp_mb1 = self.temp_mb1.sort_values(by=["statistic"], ascending=False)
        print("after remove irrelevance features, left "+str(self.temp_mb1.shape[0])+" features.")
        # display(self.temp_mb1.head(n=24))
        # Markov boundry discovery
        # Phase II: forward
        print("start phase II: forward checking")
        base_corr = 0
        for index, row in self.temp_mb1.iterrows():
            temp_mb = self.temp_mb2.append(row, ignore_index=True)
            var_list = self.col_names[temp_mb["idx"].values.astype(int)]
            dis_corr = dcor.distance_correlation(data[var_list].values, target.values, exponent=2)
            if dis_corr > base_corr:
                print("MB adds " + row["name"]+" index is "+str(index))
                base_corr = dis_corr
                self.temp_mb2 = self.temp_mb2.append(row, ignore_index=True)
        print("after forward checking, left "+str(self.temp_mb2.shape[0])+" features.")
        # display(self.temp_mb2.head(n=24))
        # Phase III: backward
        print("start phase III: backward checking")
        self.temp_mb2 = self.temp_mb2.sort_values(by=["statistic"], ascending=True)
        self.mb = self.temp_mb2.copy()
        for index, row in self.temp_mb2.iterrows():
            temp_mb = self.mb.drop(self.mb.loc[self.mb["idx"] == row["idx"]].index)
            var_list = self.col_names[temp_mb["idx"].values.astype(int)]
            dis_corr = dcor.distance_correlation(data[var_list].values, target.values, exponent=2)
            if dis_corr >= base_corr:
                print("MB deletes " + row["name"])
                base_corr = dis_corr
                self.mb.drop(self.mb.loc[self.mb["idx"] == row["idx"]].index, inplace=True)
        print("after backward checking, left "+str(self.mb.shape[0])+" features.")
        self.mb = self.mb.sort_values(by=["statistic"], ascending=False)
        return self.mb

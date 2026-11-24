from merging_features import loader,smoothness_gen
from k_imputer import knn_imputer
from feature_eng import add_time
from pathlib import Path
import pandas as pd

class Orcha:

    def __init__(self,data_input_path: Path) -> None:

        self.in_d_pth = data_input_path

        self.data = pd.read_csv(self.in_d_pth,index_col=0,parse_dates=True)

    def apply_knn_imputer(self,ind2cols : dict[int,list[str]]):

        knn_imp_object = knn_imputer.KNNImputerClass(self.in_d_pth)

        data = self.data.copy()

        return knn_imp_object.perform_knn_imputation(data,ind2cols,5)

    def make_learnable_gating(self):

        
        smoothed_data_getter = smoothness_gen.PemsRowSmoothing(self.data.copy(),k=3)

        merged_loader = loader.PEMSForecastDataset(self.data.copy(),smoothed_data_getter.transform()) # this is pytorch loader

        return merged_loader

    def add_time_feature(self,cyclical=True):

        feature_object = add_time.TimeFeatureEngineer()

        if cyclical == True:
            return feature_object.transform(self.data.copy())
        else:
            return feature_object.transform(self.data.copy(),'normalized')

    def define_your_pipeline(self):
        pass

        
if __name__ == '__main__':

    q = Orcha(Path('PEMS-BAY.csv'))

    qq = q.add_time_feature()

    print(qq.columns)
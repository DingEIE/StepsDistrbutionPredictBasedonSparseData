import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    file_withoutweek = "./GRUPredictWithoutWeek2101/results_every_ID_withoutweek.csv"
    file_withweek = "./GRUPredictwithWeek2101/results_every_ID_withweek.csv"
    file_workingday = "./GRUPredictwithWorkingDay2101/results_every_ID_withworkingday.csv"
    df_withoutweek = pd.read_csv(file_withoutweek).set_index('ID')
    df_withoutweek.columns = ['WithoutWeek_SMAPE', 'WithoutWeek_RMSE', 'WithoutWeek_R2', 'WithoutWeek_R']
    df_withweek = pd.read_csv(file_withweek).set_index('ID')
    df_withweek.columns = ['WithWeek_SMAPE', 'WithWeek_RMSE', 'WithWeek_R2', 'WithWeek_R']
    df_workingday = pd.read_csv(file_workingday).set_index('ID')
    df_workingday.columns = ['WorkingDay_SMAPE', 'WorkingDay_RMSE', 'WorkingDay_R2', 'WorkingDay_R']
    df1 = pd.merge(df_withweek, df_withoutweek, how='left', on='ID')
    df_all = pd.merge(df1, df_workingday, how='left', on='ID')
    order = ["WithWeek_SMAPE", "WithoutWeek_SMAPE", "WorkingDay_SMAPE", "WithWeek_RMSE", "WithoutWeek_RMSE", "WorkingDay_RMSE", "WithWeek_R2", "WithoutWeek_R2", "WorkingDay_R2", 'WithoutWeek_R', 'WithWeek_R', 'WorkingDay_R']
    df_all = df_all[order]
    # map_ = {0: "With week", 1: "Without week", 2: "Working Day"}
    df_all['best_SMAPE'] = df_all.iloc[:, 0:3].idxmin(axis=1).apply(lambda x: x.replace("_SMAPE", ''))
    # .apply(lambda x: map_[x])
    df_all['best_SMAPE_val']= df_all.iloc[:, 0:3].min(axis=1)

    df_all['best_RMSE'] = df_all.iloc[:, 3:6].idxmin(axis=1).apply(lambda x: x.replace("_RMSE", ''))
    df_all['best_RMSE_val']= df_all.iloc[:, 3:6].min(axis=1)

    df_all['best_R2'] = df_all.iloc[:, 6:9].idxmax(axis=1).apply(lambda x: x.replace("_R2", ''))
    df_all['best_R2_val']= df_all.iloc[:, 6:9].max(axis=1)

    df_all['best_R'] = df_all.iloc[:, 9:12].idxmax(axis=1).apply(lambda x: str(x).replace("_R", ''))
    df_all['best_R_val']= df_all.iloc[:, 9:12].max(axis=1)

    df_all.to_csv("./Compare_Results.csv")

if __name__ == "__main__":
    main()


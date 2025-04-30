import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *


AllSamples = pd.read_csv(r"path/to/your/SCAN_USCRN_HLSL30_0.1_ts_interpolated_timeseries.csv")
root = Path(r"root/path/to/your/checkpoint")
out_dir = Path(r'dir/to/save/the/plot')

stations = ['USCRN_St._Mary_1_SSW', 'USCRN_Sandstone_6_W', 'USCRN_Cortez_8_SE', 'SCAN_N_Piedmont_Arec',
            'USCRN_Stovepipe_Wells_1_SW', 'SCAN_UAPB_Marianna', 'SCAN_Knox_City', 'USCRN_Everglades_City_5_NE']

for year in [2019, 2020, 2021, 2022]:
    for LC in AllSamples.LC_name.unique():
        (out_dir / str(year) / LC).mkdir(exist_ok=True, parents=True)


def load_result(model):
    model_dir = root / rf"SCAN_USCRN_HLSL30_0.1_{model}_exp2"
    for fold in range(1, 6):
        test_df = pd.read_csv(model_dir / f'Fold_{fold}' / f'y_test_pred_ts.csv', index_col=0)

        if fold == 1:
            result_df = test_df
        else:
            result_df = pd.concat([result_df, test_df]).reset_index(drop=True)
    return result_df

for year in [2019]:

    mmnet_result = load_result('mmnet')
    mlp_result = load_result('mlp')
    trfm_result = load_result('transformer')

    data = AllSamples[AllSamples.Year == year].reset_index(drop=True)
    for station_name in stations:
        station = data[data.ID == station_name]
        print(station_name)
        station_result1 = mmnet_result[(mmnet_result.ID == station_name) * (mmnet_result.Date.apply(lambda x: str(x).startswith(str(year))))]
        station_result2 = mlp_result[(mlp_result.ID == station_name) * (mlp_result.Date.apply(lambda x: str(x).startswith(str(year))))]
        station_result3 = trfm_result[(trfm_result.ID == station_name) * (trfm_result.Date.apply(lambda x: str(x).startswith(str(year))))]

        test_true = station_result1.y_test.values
        test_pred1 = station_result1.y_pred.values
        test_pred2 = station_result2.y_pred.values
        test_pred3 = station_result3.y_pred.values

        years = station.Year.values
        doys = pd.to_datetime(station.Date.apply(str)).apply(lambda x: x.day_of_year).values
        SMAP = station.SMAP
        prec = station.pr

        ################################################
        #                     plot                     #
        ################################################
        fig, ax = plt.subplots(figsize=(8, 3))

        in_situ = ax.scatter(doys, test_true, s=2, label='In-Situ SM')  # scatter - s=2
        pred1 = ax.plot(doys, test_pred1, linewidth=1.5, label='MMNet')
        pred2 = ax.plot(doys, test_pred2, linewidth=1, alpha=0.8, label='MLP')
        pred3 = ax.plot(doys, test_pred3, linewidth=1, alpha=0.8, label='Transformer')
        smap = ax.plot(doys, SMAP, '--', linewidth=1, color='gray', alpha=0.5, label='SMAP SM')

        ax.set_ylabel('Soil Moisture (m\u00b3/m\u00b3)')
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                      [f'{year}-{mon}' for mon in range(1, 13)], rotation=30)
        ax.set_xlim(0, 366)
        ax.set_ylim(0, 0.6)

        ax2 = ax.twinx()
        pr = ax2.bar(doys, prec, color='lightblue', label='Precipitation')
        ax2.invert_yaxis()
        ax2.set_ylabel('Precipitation (mm)')
        ax2.set_ylim(100, 0)

        lb = [in_situ] + pred1 + pred2 + pred3 + smap + [pr]
        names = [l.get_label() for l in lb]
        # if station_name in ['USCRN_Stovepipe_Wells_1_SW']:
        #     ax.legend(lb, names, loc=2)
        # elif station_name in ['USCRN_Everglades_City_5_NE']:
        #     ax.legend(lb, names, loc=4)
        # else:
        #     ax.legend(lb, names, loc=3)
        plt.title(f"{station_name}, LC={station.LC_name.iloc[0]}")
        plt.tight_layout()

        ################################################
        #                  end plot                    #
        ################################################
        # plt.show()
        plt.savefig(out_dir / f"{year}" / f"{station.LC_name.iloc[0]}_ext.png", dpi=600) #/ f'{station.LC_name.iloc[0]}.png')
        plt.close()

        # scores1 = accuracy(test_true, test_pred1)
        # scores2 = accuracy(test_true, test_pred2)
        # scores3 = accuracy(test_true, test_pred3)
        # scores_smap = accuracy(test_true, SMAP.values)
        #
        # print("\t".join(str(value) for value in scores1.values()), end='\t')
        # print("\t".join(str(value) for value in scores2.values()), end='\t')
        # print("\t".join(str(value) for value in scores3.values()), end='\t')
        # print("\t".join(str(value) for value in scores_smap.values()), end='\t')
        # print()


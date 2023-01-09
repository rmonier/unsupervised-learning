import matplotlib.pyplot as plt
import pathlib as pl
import pandas as pd

def main():
    # Get smallest MAPE error for all datasets
    for filename_overview in pl.Path("evaluation/results").glob('som_*.csv'):
        print(f"Processing {filename_overview}...")
        with open(filename_overview, 'r') as f_overview:
            df = pd.read_csv(filename_overview, encoding='utf8', sep=',')
            min_data = df.iloc[df["MAPE Error"].idxmin()]
            print("> Min MAPE values: ", min_data["Layers"], min_data["Activation Function"], min_data["Learning Rate"], min_data["Momentum"], min_data["Epochs"], min_data["MAPE Error"])
            
            # Plot epoch error for best parameters
            min_layers = min_data["Layers"].replace("; ", "-")[1:-1]
            filename_best_paras = f"evaluation/results/epochs_{filename_overview.stem}_{min_data['Activation Function']}_{min_data['Epochs'].astype(int)}_{min_data['Learning Rate'].astype(float)}_{min_data['Momentum'].astype(float)}_{min_layers}.csv"
            print("> Best Parameters quadratic error progression: " + filename_best_paras)

            df = pd.read_csv(filename_best_paras, encoding='utf8', sep=',')
            df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], color='red', title=f"Epochs vs MAPE Error for {filename_overview.stem} best parameters")

            # Plot data for each pattern of the last epoch for best parameters
            filename_pattern_best_paras = f"evaluation/results/eval_{filename_overview.stem}_{min_data['Activation Function']}_{min_data['Epochs'].astype(int)}_{min_data['Learning Rate'].astype(float)}_{min_data['Momentum'].astype(float)}_{min_layers}.csv"
            print("> Best Parameters pattern evaluation progression: " + filename_pattern_best_paras)

            df_pattern = pd.read_csv(filename_pattern_best_paras, encoding='utf8', sep=',')
            df_pattern.plot(kind='scatter', x=df_pattern.columns[0], y=df_pattern.columns[1], color='blue', title=f"Z vs Y for {filename_overview.stem} best parameters")
    plt.show()

if __name__ == '__main__':
    main()
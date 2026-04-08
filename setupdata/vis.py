import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_load_patterns(input_file="caiso_features.csv"):
    df = pd.read_csv(input_file)
    df['CAISO'] = pd.to_numeric(df['CAISO'], errors='coerce')
    df = df.dropna(subset=['CAISO'])

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=df, x='hour', y='CAISO', hue='is_weekend', marker='o')

    plt.title('Average CAISO Load: Weekdays (0) vs Weekends (1)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (MW)')
    plt.xticks(range(1, 25))

    plt.axvspan(16, 21, color='orange', alpha=0.2, label='Peak Window')
    plt.legend()

    plt.savefig('load_patterns.png')
    plt.show()


if __name__ == "__main__":
    visualize_load_patterns()

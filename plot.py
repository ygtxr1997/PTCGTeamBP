import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_decks_win_rates(
    ax: plt.Axes,
    deck_names: list = ('Lugia Archeops', 'Mew Genesect', 'Lost Zone Box', 'Regis', 'Goodra LZ Box', 'Rayquaza LZ Box'),
    win_rates: list = (53.11933333333334, 53.717999999999996, 51.44466666666666, 52.530666666666676, 53.62733333333333,
        52.364666666666665),
    title: str = "Deck Win Rates (Sorted)",
):
    """
    Plots a heatmap of deck win rates on the provided Matplotlib axis.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        deck_names (list): List of deck names.
        win_rates (list): List of win rates corresponding to the deck names.

    Returns:
        plt.Axes: The Matplotlib axis with the heatmap plotted.
    """
    # Reorder data by win rate (descending order)
    sorted_data = sorted(zip(win_rates, deck_names), reverse=True)
    win_rates_sorted, deck_names_sorted = zip(*sorted_data)  # Unzip the sorted data

    # Reshape the data to fit a heatmap (1 row for this case)
    data = np.array(win_rates_sorted).reshape(1, -1)

    # Create the heatmap on the provided axis
    block_labels = [label.replace(" ", "\n") for label in deck_names_sorted]
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=block_labels,
        yticklabels=["Win Rate"],
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )

    # Customize axis labels
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='x', rotation=0)  # Rotate x-axis labels for better readability
    ax.tick_params(axis='y')

    return ax


def plot_win_rate_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    our_decks: list,
    opponent_decks: list,
    title: str = "Win Rate Matrix Heatmap",
):
    """
    Plots a heatmap for a 6x6 win rate matrix on the provided Matplotlib axis.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        matrix (np.ndarray): A 6x6 matrix of win rates.
        our_decks (list): List of names for our decks (x-axis).
        opponent_decks (list): List of names for opponent decks (y-axis).

    Returns:
        plt.Axes: The Matplotlib axis with the heatmap plotted.
    """
    if not isinstance(ax, plt.Axes):
        raise ValueError("The 'ax' parameter must be a valid Matplotlib Axes object.")

    m, n = matrix.shape

    matrix_with_mean = np.zeros((m + 1, n + 1), dtype=int)
    col_avg = matrix.mean(axis=1)
    row_avg = matrix.mean(axis=0)
    all_avg = col_avg.mean()
    matrix_with_mean[:m, :n] = matrix
    matrix_with_mean[:m, n] = col_avg
    matrix_with_mean[m:, :n] = row_avg
    matrix_with_mean[m, n] = all_avg

    # Plot the heatmap
    block_labels = [label.replace(" ", "\n") for label in opponent_decks + ['One vs. All (Avg.)']]
    sns.heatmap(
        matrix_with_mean,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=block_labels,
        yticklabels=our_decks + ['All vs. One (Avg.)'],
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )

    # Customize axis labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Opponent Decks", fontsize=12)
    ax.set_ylabel("Our Decks", fontsize=12)
    ax.tick_params(axis='x', rotation=0)  # Rotate x-axis labels
    ax.tick_params(axis='y', rotation=0)              # Keep y-axis labels horizontal

    return ax


def plot_pick_comb_avg_top2(
        ax: plt.Axes,
        pick_combs: list = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        avg_win_rates: list = (54.17, 53.68, 54.41, 55.488, 56.22, 55.73, 54.68, 55.417, 54.92, 56.73),
        top2_win_rates: list = (57.37, 56.50, 59.64, 58.16, 61.29, 60.43, 57.1885, 60.3205, 59.45, 61.11),
        title: str = "Picking Combinations' Avg and Top-2 Win Rates (Sorted)",
):  # 2x6
    # Combine data into rows and sort by average win rate (descending order)
    combined_data = list(zip(pick_combs, avg_win_rates, top2_win_rates))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)  # Sort by avg_win_rates

    # Extract sorted values
    sorted_combs, sorted_avg, sorted_top2 = zip(*sorted_data)

    # Prepare the data for the heatmap (2 rows: Avg and Top-2)
    heatmap_data = np.array([sorted_avg, sorted_top2])  # Shape: (2, len(pick_combs))

    # Format combination labels in block style
    sorted_combs_adjusted = [(c[0] + 1, c[1] + 1, c[2] + 1) for c in sorted_combs]
    comb_labels = [f"({', '.join(map(str, comb))})" for comb in sorted_combs_adjusted]  # Convert tuples to strings

    # Add newline after each element for block-style labels
    block_labels = [label.replace(", ", ",\n") for label in comb_labels]

    # Create the heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=block_labels,
        yticklabels=["Avg Win%", "Top2 Win%"],
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )

    # Customize axis labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Picking Combinations of Decks (选用卡组组合)", fontsize=12)
    # ax.set_ylabel("Metrics", fontsize=12)
    ax.tick_params(axis='x', rotation=0)  # No rotation since combinations are block-style
    ax.tick_params(axis='y', rotation=0)  # Rotate x-axis labels

    return ax


if __name__ == '__main__':
    fig = plt.figure(figsize=(16, 12))  # Set the overall figure size
    grid = fig.add_gridspec(4, 2)  # Create a 6x2 grid

    # Axes for the first column (4 figures)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[3, 0]),
    ]

    # Axes for the second column (2 figures)
    axes += [
        fig.add_subplot(grid[:2, 1]),  # Top 3 rows combined into 1 plot
        fig.add_subplot(grid[2:, 1])  # Bottom 2 rows combined into 1 plot
    ]

    ''' 1x6 '''
    for i in range(4):
        plot_decks_win_rates(axes[i])

    ''' 6x6 '''
    matrix = np.array([
        [53.1, 52.4, 50.8, 54.3, 51.9, 53.2],
        [52.7, 53.9, 51.5, 52.6, 54.1, 50.3],
        [51.4, 52.2, 53.6, 51.8, 52.5, 54.0],
        [53.3, 51.9, 52.8, 53.1, 51.7, 52.4],
        [50.5, 51.6, 53.0, 54.2, 52.3, 52.8],
        [54.0, 53.2, 51.7, 52.9, 50.8, 54.1]
    ])
    our_decks = ['Lugia', 'Mew', 'Lost Box', 'Regis', 'Goodra', 'Rayquaza']
    opponent_decks = ['Shadow', 'Blissey', 'Dialga', 'Giratina', 'Kyurem', 'Darkrai']
    plot_win_rate_matrix(axes[4], matrix, our_decks, opponent_decks)
    plot_win_rate_matrix(axes[5], matrix, our_decks, opponent_decks)

    plt.tight_layout()
    plt.show()



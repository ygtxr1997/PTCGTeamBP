import os
from functools import partial

import gradio as gr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from itertools import combinations

from models import Team
from plot import plot_decks_win_rates, plot_win_rate_matrix
from plot import plot_pick_comb_avg_top2
from matplotlib import rcParams
import platform


if platform.system() == "Windows":
    default_font = "SimHei"
else:  # Linux 或其他系统
    default_font = "Noto Sans CJK SC"
rcParams['font.sans-serif'] = "SimHei"
rcParams['axes.unicode_minus'] = False   # 解决负号 "-" 显示为方块的问题


def read_rank_data(csv_data_dir: str = "crawl/output",
                   csv_fn: str = "00_rank_data.csv"
                   ):
    csv_file = os.path.join(csv_data_dir, csv_fn)

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return None

    return df


# Gradio components for the banning tab
def run_predict_for_banning(ours_num, ours1, ours2, ours3, ours4, ours5, ours6,
                            oppo_num, oppo1, oppo2, oppo3, oppo4, oppo5, oppo6,
                            ours1_dis, ours2_dis, ours3_dis, ours4_dis, ours5_dis, ours6_dis,
                            oppo1_dis, oppo2_dis, oppo3_dis, oppo4_dis, oppo5_dis, oppo6_dis,
                            custom_win_rate_matrix,
                            ):
    """Handles the banning tab."""
    print(ours_num, ours1, ours2, ours3, ours4, ours5, ours6,)
    print(oppo_num, oppo1, oppo2, oppo3, oppo4, oppo5, oppo6,)

    ours_decks = [ours1, ours2, ours3, ours4, ours5, ours6][:ours_num]
    ours_dis = [ours1_dis, ours2_dis, ours3_dis, ours4_dis, ours5_dis, ours6_dis][:ours_num]
    ours_team = Team(ours_decks, win_rate_discounts=[float(x) for x in ours_dis])

    oppo_decks = [oppo1, oppo2, oppo3, oppo4, oppo5, oppo6][:oppo_num]
    oppo_dis = [oppo1_dis, oppo2_dis, oppo3_dis, oppo4_dis, oppo5_dis, oppo6_dis][:oppo_num]
    oppo_team = Team(oppo_decks, win_rate_discounts=[float(x) for x in oppo_dis])

    print(ours_dis)
    print(oppo_dis)

    matrix_6x6 = ours_team.vs_team(oppo_team)
    custom_win_rate_matrix = custom_win_rate_matrix.iloc[:, 1:].to_numpy().astype(float)
    print(custom_win_rate_matrix)

    print([x.limit_id for x in ours_team.decks])
    out_str = []

    ''' 1. Without customization matrix '''
    # Opponent decides to ban our decks
    banning_our_deck, banning_our_wr, all_banning_our_wr = oppo_team.best_ban_policy(ours_team)
    out_str.append((f"Opponent Will Ban Our Deck: {ours_team.decks[banning_our_deck].deck_name}({banning_our_deck}), "
                    f"Avg.Win%={banning_our_wr:.2f}"))

    print(ours_decks)
    print(all_banning_our_wr)

    # Based on predicted opponent's banning deck, we decide to ban opponent's deck
    banning_opponent_deck, banning_wr, all_banning_wr = ours_team.best_ban_policy(oppo_team)
    out_str.append((f"We Will Ban Opponent Deck: {oppo_team.decks[banning_opponent_deck].deck_name}({banning_opponent_deck}), "
                    f"Avg.Win%={banning_wr:.2f}"))

    print(all_banning_wr)

    ''' 2. With customization matrix '''
    # Opponent decides to ban our decks
    banning_our_deck, banning_our_wr, custom_all_banning_our_wr = oppo_team.best_ban_policy(
        ours_team, win_rate_matrix=custom_win_rate_matrix.transpose())
    out_str.append((f"Opponent Will Ban Our Deck: {ours_team.decks[banning_our_deck].deck_name}({banning_our_deck}), "
                    f"Avg.Win%={banning_our_wr:.2f}"))

    # Based on predicted opponent's banning deck, we decide to ban opponent's deck
    banning_opponent_deck, banning_wr, custom_all_banning_wr = ours_team.best_ban_policy(
        oppo_team, win_rate_matrix=custom_win_rate_matrix)
    out_str.append(
        (f"We Will Ban Opponent Deck: {oppo_team.decks[banning_opponent_deck].deck_name}({banning_opponent_deck}), "
         f"Avg.Win%={banning_wr:.2f}"))

    # Plot
    fig = plt.figure(figsize=(24, 12))  # Set the overall figure size
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
    plt_ours_decks = [f"{x}({i + 1})" for i, x in enumerate(ours_decks)]
    plt_oppo_decks = [f"{x}({i + 1})" for i, x in enumerate(oppo_decks)]
    plot_decks_win_rates(axes[0], plt_ours_decks, all_banning_our_wr,
                         title="If Opponents Ban Our Deck (by Internet Data)\n(假设对手禁用我方卡组(参考国际数据))")
    plot_decks_win_rates(axes[1], plt_oppo_decks, all_banning_wr,
                         title="If We Ban Opponent Deck (by Internet Data)\n(假设我方禁用对手卡组(参考国际数据))")
    plot_decks_win_rates(axes[2], plt_ours_decks, custom_all_banning_our_wr,
                         title="If Opponents Ban Our Deck (by Customized Data)\n(假设对手禁用我方卡组(参考自定义胜率))")
    plot_decks_win_rates(axes[3], plt_oppo_decks, custom_all_banning_wr,
                         title="If We Ban Opponent Deck (by Customized Data)\n(假设我方禁用对手卡组(参考自定义胜率))")

    ''' 6x6 '''
    plot_win_rate_matrix(axes[4], matrix_6x6, plt_ours_decks, plt_oppo_decks,
                         title="Win% Of Ours vs. Opponents (by Internet Data)\n(我方卡组vs对手卡组胜率详情(参考国际数据))")
    plot_win_rate_matrix(axes[5], custom_win_rate_matrix, plt_ours_decks, plt_oppo_decks,
                         title="Win% Of Ours vs. Opponents (by Customized Data)\n(我方卡组vs对手卡组胜率详情(参考自定义胜率))")

    plt.tight_layout()

    print("\n".join(out_str))
    return plt


def get_picking_policy_after_banning(
        ours_num, ours1, ours2, ours3, ours4, ours5, ours6, ours_banned,
        oppo_num, oppo1, oppo2, oppo3, oppo4, oppo5, oppo6, oppo_banned,
        ours1_dis, ours2_dis, ours3_dis, ours4_dis, ours5_dis, ours6_dis,
        oppo1_dis, oppo2_dis, oppo3_dis, oppo4_dis, oppo5_dis, oppo6_dis,
        custom_win_rate_matrix,
):
    print(ours_num, ours1, ours2, ours3, ours4, ours5, ours6, )
    print(oppo_num, oppo1, oppo2, oppo3, oppo4, oppo5, oppo6, )

    ours_decks = [ours1, ours2, ours3, ours4, ours5, ours6][:ours_num]
    ours_dis = [ours1_dis, ours2_dis, ours3_dis, ours4_dis, ours5_dis, ours6_dis][:ours_num]
    ours_team = Team(ours_decks, win_rate_discounts=[float(x) for x in ours_dis])

    oppo_decks = [oppo1, oppo2, oppo3, oppo4, oppo5, oppo6][:oppo_num]
    oppo_dis = [oppo1_dis, oppo2_dis, oppo3_dis, oppo4_dis, oppo5_dis, oppo6_dis][:oppo_num]
    oppo_team = Team(oppo_decks, win_rate_discounts=[float(x) for x in oppo_dis])

    custom_win_rate_matrix = custom_win_rate_matrix.iloc[:, 1:].to_numpy().astype(float)

    ours_banned = int(ours_banned.split(' ')[0]) - 1
    oppo_banned = int(oppo_banned.split(' ')[0]) - 1
    print(ours_banned, oppo_banned)

    ''' 1. Without customization matrix '''
    # Find the best picking policy for opponent
    best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick, oppo_verbose_out = oppo_team.best_picking_policy(
        ours_team, pick_size=3, self_banned=oppo_banned, opponent_banned=ours_banned, return_verbose=True,
    )
    opponents_picking = [best_avg_pick, best_top2_pick]
    print("Opponent Best Average Win Rate:", best_avg_wr, "Top-2 Win Rate:", best_top2_wr)

    # Find the best picking policy for our team
    # opponents_picking = None
    best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick, ours_verbose_out = \
        ours_team.best_picking_policy(
            oppo_team, pick_size=3,
            opponents_pick_policy=opponents_picking,
            self_banned=ours_banned,
            opponent_banned=oppo_banned,
            return_verbose=True,
        )
    print("Best Average Win Rate:", best_avg_wr)
    print("Best Average Team Pick:", [ours_decks[i] for i in best_avg_pick])
    print("Best Top-2 Win Rate:", best_top2_wr)
    print("Best Top-2 Team Pick:", [ours_decks[i] for i in best_top2_pick])

    ''' 2. With customization matrix '''
    # Find the best picking policy for opponent
    best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick, custom_oppo_verbose_out = oppo_team.best_picking_policy(
        ours_team, pick_size=3, self_banned=oppo_banned, opponent_banned=ours_banned, return_verbose=True,
        win_rate_matrix=custom_win_rate_matrix.transpose(),
    )
    opponents_picking = [best_avg_pick, best_top2_pick]
    print("Opponent Best Average Win Rate:", best_avg_wr, "Top-2 Win Rate:", best_top2_wr)

    # Find the best picking policy for our team
    # opponents_picking = None
    best_avg_wr, best_avg_pick, best_top2_wr, best_top2_pick, custom_ours_verbose_out = \
        ours_team.best_picking_policy(
            oppo_team, pick_size=3,
            opponents_pick_policy=opponents_picking,
            self_banned=ours_banned,
            opponent_banned=oppo_banned,
            return_verbose=True,
            win_rate_matrix=custom_win_rate_matrix,
        )
    print("Best Average Win Rate:", best_avg_wr)
    print("Best Average Team Pick:", [ours_decks[i] for i in best_avg_pick])
    print("Best Top-2 Win Rate:", best_top2_wr)
    print("Best Top-2 Team Pick:", [ours_decks[i] for i in best_top2_pick])

    # Plot
    fig = plt.figure(figsize=(12, 8))  # Set the overall figure size
    grid = fig.add_gridspec(4, 1)  # Create a 6x2 grid

    # Axes for the first column (4 figures)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[3, 0]),
    ]

    ''' 2xC(5,2) '''
    plot_pick_comb_avg_top2(axes[0], oppo_verbose_out['combinations'],
                            oppo_verbose_out['pick_avg_wrs'],
                            oppo_verbose_out['pick_top_wrs'],
                            title="Opponents Picking Their Decks (by Internet Data)\n(对手选用不同卡组的胜率(参考国际数据))")
    plot_pick_comb_avg_top2(axes[1], ours_verbose_out['combinations'],
                            ours_verbose_out['pick_avg_wrs'],
                            ours_verbose_out['pick_top_wrs'],
                            title="Our Team Picking Our Decks (by Internet Data)\n(我方选用不同卡组的胜率(参考国际数据))")
    plot_pick_comb_avg_top2(axes[2], custom_oppo_verbose_out['combinations'],
                            custom_oppo_verbose_out['pick_avg_wrs'],
                            custom_oppo_verbose_out['pick_top_wrs'],
                            title="Opponents Picking Their Decks (by Customized Data)\n(对手选用不同卡组的胜率(参考自定义胜率))")
    plot_pick_comb_avg_top2(axes[3], custom_ours_verbose_out['combinations'],
                            custom_ours_verbose_out['pick_avg_wrs'],
                            custom_ours_verbose_out['pick_top_wrs'],
                            title="Our Team Picking Our Decks (by Customized Data)\n(我方选用不同卡组的胜率(参考自定义胜率))")

    plt.tight_layout()

    return plt


# Gradio components for the picking tab
def picking_interface(pick_size):
    """Handles the picking tab."""
    return 0
    result = compute_best_picking_policy(pick_size)
    return result


# 处理输入的函数
def process_input(option, custom_input):
    if custom_input:  # 如果用户输入自定义文本
        return f"Your input: {custom_input}"
    elif option:  # 如果用户从下拉菜单选择
        return f"Selected option: {option}"
    else:
        return "No input provided."


# 函数：根据选择返回对应图片 URL
def display_image(selected_option):
    return image_options.get(selected_option, None)


# 动态更新下拉框的可见性
def update_dropdowns(num_decks):
    """根据选定数量动态显示下拉框"""
    updates = []
    for i in range(6):
        if i < num_decks:  # 显示前 num_decks 个下拉框
            updates.append(gr.update(visible=True))
        else:  # 隐藏多余的下拉框
            updates.append(gr.update(visible=False))
    return updates


# Function to dynamically generate a win rate matrix
def generate_matrix_with_labels(ours_num, oppo_num):
    # Create a default win rate matrix
    matrix = np.full((ours_num, oppo_num), 50.0).tolist()  # Default win rate: 50.0

    # Add "Our Decks" labels in the first column
    ours_labels = [f"Our Deck {i + 1}" for i in range(ours_num)]
    matrix_with_labels = [[label] + row for label, row in zip(ours_labels, matrix)]
    return matrix_with_labels


# Update the matrix when the dropdown values change
def update_matrix_with_labels(ours_num, oppo_num):
    matrix = generate_matrix_with_labels(ours_num, oppo_num)
    headers = ["Our Decks (我方卡组)"] + [f"Opponent (对方) {i + 1}" for i in range(oppo_num)]
    return gr.update(value=matrix, headers=headers,
                     row_count=(ours_num, "fixed"),
                     col_count=(oppo_num + 1, "fixed"))


# 生成动态图像的函数
def display_figure(deck_name: str, name_to_id: dict):
    # Retrieve the folder name using the deck name
    limit_id = name_to_id[deck_name]

    # Directory containing the figures
    fig_dir = os.path.join("crawl/output/fig", limit_id)

    # Get and sort all image filenames in the folder
    fig_fns = os.listdir(fig_dir)
    fig_fns.sort()

    # Load all images
    images = []
    for fig_fn in fig_fns:
        fig_path = os.path.join(fig_dir, fig_fn)
        try:
            img = mpimg.imread(fig_path)
            images.append((img, fig_fn))  # Store image and its filename
        except Exception as e:
            print(f"Error reading {fig_path}: {e}")

    # Check if images are available
    if not images:
        raise ValueError("No images found in the folder.")

    # Define a fixed height for all figures (e.g., 4 inches), and dynamically calculate width
    fixed_height = 1  # Fixed height in inches
    num_images = len(images)
    fixed_width = 2  # Each image gets 4 inches of width

    # Create the figure with the fixed height
    fig, axes = plt.subplots(1, num_images, figsize=(fixed_width, fixed_height))

    # Ensure axes is iterable if there's only one image
    if num_images == 1:
        axes = [axes]

    # Plot each image
    for ax, (img, name) in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        # ax.set_title(name)

    plt.tight_layout()
    return fig  # Return the figure for Gradio to display


# Read rank .csv data
rank_data = read_rank_data()
deck_names = rank_data['Deck'].tolist()
limit_ids = rank_data['Limit_ID'].tolist()
deck_to_limit = {k: v for k, v in zip(deck_names, limit_ids)}


# Gradio app with two tabs
with gr.Blocks(
    css="""
    .custom-row .gr-column {
        width: 13%;  /* 每个元素占 1/6 的宽度 */
        flex: none;
    }
    .custom-image img {
        max-width: 50px; /* 设置最大宽度为 200px */
        height: auto; /* 保持宽高比 */
    }
    .fixed-plot-height {
        height: 100px; /* Enforce consistent height */
    }
    .dataframe-max-width {
        max-width: 1200px; /* Limit the max width to 800px */
        margin: 0 auto;   /* Center the dataframe */
        overflow-x: auto; /* Allow horizontal scrolling if content overflows */
    }
    .dataframe-max-width table {
        text-align: center; /* Center-align table text */
        margin: 0 auto;     /* Center-align table as a whole */
    }
    .dataframe-max-width th, .dataframe-max-width td {
        text-align: center; /* Center-align header and cell content */
        vertical-align: middle; /* Center-align vertically */
    }
    """,
    fill_width=True,
) as demo:
    gr.Markdown("## 🎴 PTCG Team Match: Ban & Pick Tools 🤩")

    with gr.Tab("Banning & Picking"):
        gr.Markdown("### 📜使用说明:")
        gr.Markdown("```shell"
                    "a. 设置我方卡组和对方卡组; (可选) 根据需要调整胜率衰减值，默认是 1.0  \n"
                    "b. 本工具默认使用国际服统计数据(https://play.limitlesstcg.com/decks?format=standard&rotation=2022&set=SIT)  \n"
                    "c. (可选) 在本页面第3.步中可以输入自定义胜率，预测时会分别显示基于国际数据和基于自定义胜率的结果  \n"
                    "d. 点击 `Run Prediction for Banning` 按钮，即可预测双方禁用结果  \n"
                    "e. 在本页面第5.步中可以输入实际禁用情况  \n"
                    "f. 点击 `Get Best Picking Policy`，即可获取最佳选用策略  \n"
                    "```"
                    )

        gr.Markdown("### 🐱1. Our Decks (设置我方卡组)")
        gr.Markdown("```Win% Discount (胜率衰减) 的相关解释：假设一个队伍有 3 套甚至 4 套洛奇亚，那么这些卡组的胜率并不能 100% 达到"
                    "国际数据的胜率，因此可以给这些卡组输入一个衰减值，胜率衰减值建议范围在 0.95~0.98 附近```")
        default_ours_num = 6
        with gr.Row(equal_height=True):
            dropdown_ours_num = gr.Dropdown(label="#Our Decks (我方卡组数量)", choices=[4, 5, 6], interactive=True,
                                            value=default_ours_num)
            options = deck_names
            display_figure_by_dict = partial(display_figure, name_to_id=deck_to_limit)
            with gr.Column():
                dropdown_ours1 = gr.Dropdown(label="Deck 1 (U)", choices=options, interactive=True)
                # deck1_image = gr.Image(label="Deck 1 (U)", interactive=False)
                deck1_image = gr.Plot(label="Deck 1 (U)", elem_classes="fixed-plot-height")
                dropdown_ours1.change(display_figure_by_dict, inputs=[dropdown_ours1], outputs=[deck1_image])
                ours1_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_ours2 = gr.Dropdown(label="Deck 2 (V)", choices=options, interactive=True)
                # deck2_image = gr.Image(label="Deck 2 (V)", interactive=False)
                deck2_image = gr.Plot(label="Deck 2 (V)", elem_classes="fixed-plot-height")
                dropdown_ours2.change(display_figure_by_dict, inputs=[dropdown_ours2], outputs=[deck2_image])
                ours2_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_ours3 = gr.Dropdown(label="Deck 3 (W)", choices=options, interactive=True)
                deck3_image = gr.Plot(label="Deck 3 (W)", elem_classes="fixed-plot-height")
                dropdown_ours3.change(display_figure_by_dict, inputs=[dropdown_ours3], outputs=[deck3_image])
                ours3_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_ours4 = gr.Dropdown(label="Deck 4 (X)", choices=options, interactive=True)
                deck4_image =gr.Plot(label="Deck 4 (X)", elem_classes="fixed-plot-height")
                dropdown_ours4.change(display_figure_by_dict, inputs=[dropdown_ours4], outputs=[deck4_image])
                ours4_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_ours5 = gr.Dropdown(label="Deck 5 (Y)", choices=options, interactive=True)
                deck5_image = gr.Plot(label="Deck 5 (Y)", elem_classes="fixed-plot-height")
                dropdown_ours5.change(display_figure_by_dict, inputs=[dropdown_ours5], outputs=[deck5_image])
                ours5_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_ours6 = gr.Dropdown(label="Deck 6 (Z)", choices=options, interactive=True)
                deck6_image = gr.Plot(label="Deck 6 (Z)", elem_classes="fixed-plot-height")
                dropdown_ours6.change(display_figure_by_dict, inputs=[dropdown_ours6], outputs=[deck6_image])
                ours6_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            dropdown_ours = [
                dropdown_ours1, dropdown_ours2, dropdown_ours3, dropdown_ours4,
                dropdown_ours5, dropdown_ours6,
            ]
            deck_image_ours = [
                deck1_image, deck2_image, deck3_image, deck4_image, deck5_image, deck6_image
            ]
            discount_ours = [
                ours1_discount, ours2_discount, ours3_discount, ours4_discount, ours5_discount, ours6_discount
            ]

            # 当选择 #Our Decks 的值时动态更新可见的下拉框
            dropdown_ours_num.change(update_dropdowns, inputs=dropdown_ours_num, outputs=dropdown_ours)
            dropdown_ours_num.change(update_dropdowns, inputs=dropdown_ours_num, outputs=deck_image_ours)
            dropdown_ours_num.change(update_dropdowns, inputs=dropdown_ours_num, outputs=discount_ours)

        gr.Markdown("### 👊2. Opponent Decks (设置对手卡组)")
        with gr.Row(equal_height=True):
            default_oppo_num = 6
            dropdown_oppo_num = gr.Dropdown(label="#Opponent Decks (对方卡组数量)", choices=[4, 5, 6], interactive=True,
                                            value=default_oppo_num)
            with gr.Column():
                dropdown_oppo1 = gr.Dropdown(label="Deck 1 (U)", choices=options, interactive=True)
                op_deck1_image = gr.Plot(label="Deck 1 (U)", elem_classes="fixed-plot-height")
                dropdown_oppo1.change(display_figure_by_dict, inputs=[dropdown_oppo1], outputs=[op_deck1_image])
                oppo1_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_oppo2 = gr.Dropdown(label="Deck 2 (V)", choices=options, interactive=True)
                op_deck2_image = gr.Plot(label="Deck 2 (V)", elem_classes="fixed-plot-height")
                dropdown_oppo2.change(display_figure_by_dict, inputs=[dropdown_oppo2], outputs=[op_deck2_image])
                oppo2_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_oppo3 = gr.Dropdown(label="Deck 3 (W)", choices=options, interactive=True)
                op_deck3_image = gr.Plot(label="Deck 3 (W)", elem_classes="fixed-plot-height")
                dropdown_oppo3.change(display_figure_by_dict, inputs=[dropdown_oppo3], outputs=[op_deck3_image])
                oppo3_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_oppo4 = gr.Dropdown(label="Deck 4 (X)", choices=options, interactive=True)
                op_deck4_image = gr.Plot(label="Deck 4 (X)", elem_classes="fixed-plot-height")
                dropdown_oppo4.change(display_figure_by_dict, inputs=[dropdown_oppo4], outputs=[op_deck4_image])
                oppo4_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_oppo5 = gr.Dropdown(label="Deck 5 (Y)", choices=options, interactive=True)
                op_deck5_image = gr.Plot(label="Deck 5 (Y)", elem_classes="fixed-plot-height")
                dropdown_oppo5.change(display_figure_by_dict, inputs=[dropdown_oppo5], outputs=[op_deck5_image])
                oppo5_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            with gr.Column():
                dropdown_oppo6 = gr.Dropdown(label="Deck 6 (Z)", choices=options, interactive=True)
                op_deck6_image = gr.Plot(label="Deck 6 (Z)", elem_classes="fixed-plot-height")
                dropdown_oppo6.change(display_figure_by_dict, inputs=[dropdown_oppo6], outputs=[op_deck6_image])
                oppo6_discount = gr.Slider(label="Win% Discount (胜率衰减)", minimum=1e-3, maximum=1., step=0.01, value=1,
                                           interactive=True)
            dropdown_oppo = [
                dropdown_oppo1, dropdown_oppo2, dropdown_oppo3, dropdown_oppo4,
                dropdown_oppo5, dropdown_oppo6,
            ]
            deck_image_oppo = [
                op_deck1_image, op_deck2_image, op_deck3_image, op_deck4_image, op_deck5_image, op_deck6_image
            ]
            discount_oppo = [
                oppo1_discount, oppo2_discount, oppo3_discount, oppo4_discount, oppo5_discount, oppo6_discount
            ]

            # 当选择 #Our Decks 的值时动态更新可见的下拉框
            dropdown_oppo_num.change(update_dropdowns, inputs=dropdown_oppo_num, outputs=dropdown_oppo)
            dropdown_oppo_num.change(update_dropdowns, inputs=dropdown_oppo_num, outputs=deck_image_oppo)
            dropdown_oppo_num.change(update_dropdowns, inputs=dropdown_oppo_num, outputs=discount_oppo)

        gr.Markdown("### 🧬3. Customizing Win Rate Matrix (Ours vs. Opponents) (自定义胜率矩阵 (我方所有 vs. 对方所有))")
        matrix_headers = ["Our Decks (我方)"] + [f"Opponent (对方) {i + 1}" for i in range(default_oppo_num)]
        customize_matrix_input = gr.Dataframe(
            label="Customized Win Rate Matrix (%)",
            value=generate_matrix_with_labels(default_ours_num, default_oppo_num),
            headers=matrix_headers,
            datatype="number",
            interactive=True,
            row_count=(default_ours_num, "fixed"),
            col_count=(default_oppo_num + 1, "fixed"),
            elem_classes="dataframe-max-width",
        )
        dropdown_ours_num.change(
            update_matrix_with_labels,
            inputs=[dropdown_ours_num, dropdown_oppo_num],
            outputs=customize_matrix_input
        )
        dropdown_oppo_num.change(
            update_matrix_with_labels,
            inputs=[dropdown_ours_num, dropdown_oppo_num],
            outputs=customize_matrix_input
        )

        gr.Markdown("### 🔮4. Run Prediction for Banning (开始预测禁用结果)")
        # row_input = gr.Number(label="Row to Ban (Index)", value=0, precision=0)
        # col_input = gr.Number(label="Column to Ban (Index)", value=0, precision=0)
        ban_button = gr.Button("Run Prediction for Banning (开始预测禁用结果)", variant="primary")
        # ban_output = gr.Textbox(label="Modified Win Rate Matrix")
        ban_output = gr.Plot(label="Prediction Result (预测结果)")
        ban_button.click(
            run_predict_for_banning,
            inputs=[
                dropdown_ours_num,
                dropdown_ours1, dropdown_ours2, dropdown_ours3, dropdown_ours4, dropdown_ours5, dropdown_ours6,
                dropdown_oppo_num,
                dropdown_oppo1, dropdown_oppo2, dropdown_oppo3, dropdown_oppo4, dropdown_oppo5, dropdown_oppo6,
                ours1_discount, ours2_discount, ours3_discount, ours4_discount, ours5_discount, ours6_discount,
                oppo1_discount, oppo2_discount, oppo3_discount, oppo4_discount, oppo5_discount, oppo6_discount,
                customize_matrix_input,
            ],
            outputs=ban_output
        )

        gr.Markdown("### 🔒5. Banning (实际禁用情况)")
        with gr.Row():
            ban_options = ["1 (U)", "2 (V)", "3 (W)", "4 (X)", "5 (Y)", "6 (Z)"]
            dropdown_banned_ours = gr.Dropdown(label="Our Banned Deck (我方被禁卡组)", choices=ban_options, interactive=True)
            dropdown_banned_oppo = gr.Dropdown(label="Opponent Banned Deck (对方被禁卡组)", choices=ban_options, interactive=True)

        gr.Markdown("### 🌠6. Get Best Picking Policy (获取最佳选用策略)")
        pick_button = gr.Button("Get Best Picking Policy (获取最佳选用策略)", variant="primary")
        pick_output = gr.Plot(label="Best Picking Policy (最佳选用策略结果)")
        pick_button.click(
            get_picking_policy_after_banning,
            inputs=[
                dropdown_ours_num,
                dropdown_ours1, dropdown_ours2, dropdown_ours3, dropdown_ours4, dropdown_ours5, dropdown_ours6,
                dropdown_banned_ours,
                dropdown_oppo_num,
                dropdown_oppo1, dropdown_oppo2, dropdown_oppo3, dropdown_oppo4, dropdown_oppo5, dropdown_oppo6,
                dropdown_banned_oppo,
                ours1_discount, ours2_discount, ours3_discount, ours4_discount, ours5_discount, ours6_discount,
                oppo1_discount, oppo2_discount, oppo3_discount, oppo4_discount, oppo5_discount, oppo6_discount,
                customize_matrix_input,
            ],
            outputs=pick_output
        )


    # with gr.Tab("Picking"):
    #     gr.Markdown("### Compute Best Picking Policy")
    #     pick_size_input = gr.Number(label="Number of Decks to Pick", value=3, precision=0)
    #     pick_button = gr.Button("Compute Best Policy")
    #     pick_output = gr.Textbox(label="Best Picking Result")
    #     pick_button.click(picking_interface, inputs=[pick_size_input], outputs=pick_output)

    with gr.Tab("Debug"):
        gr.Markdown("### Debug")
        # 输入组件
        # 图片选项及对应的 URL
        image_options = {
            "Lugia": "https://limitlesstcg.s3.us-east-2.amazonaws.com/pokemon/gen9/lugia.png",
            "Mew": "https://limitlesstcg.s3.us-east-2.amazonaws.com/pokemon/gen9/mew.png",
            "Charizard": "https://limitlesstcg.s3.us-east-2.amazonaws.com/pokemon/gen9/charizard.png"
        }

        # 下拉菜单或单选按钮
        with gr.Row():
            dropdown = gr.Dropdown(label="选择一个选项", choices=options, interactive=True)
            custom_input = gr.Textbox(label="或者输入自定义内容", placeholder="输入你的选项...")

        # 图片展示区域
        image_output = gr.Image(label="对应的图片", interactive=False)

        # 当选择下拉选项时动态更新图片
        dropdown.change(display_image, inputs=dropdown, outputs=image_output)

        submit_button = gr.Button("提交")
        output = gr.Textbox(label="输出结果")

        # 点击事件：根据输入返回结果
        submit_button.click(process_input, inputs=[dropdown, custom_input], outputs=output)

        # # 输入组件
        # dropdown_input = gr.Dropdown(label="Choose an option", choices=options, interactive=True)
        # text_input = gr.Textbox(label="Or enter custom text", placeholder="Type your input here...",
        #                         interactive=True)
        #
        # # 按钮和输出
        # submit_button = gr.Button("Submit")
        # output = gr.Textbox(label="Output")
        #
        # # 点击事件
        # submit_button.click(process_input, inputs=[dropdown_input, text_input], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8866)

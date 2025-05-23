import matplotlib.pyplot as plt
import pandas as pd

def set_parameters() -> None:
    FONT_SIZE = 20
    TICK_SIZE = 20
    # LEGEND_FONT_SIZE = 15
    LEGEND_FONT_SIZE = 20


    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=42)  # fontsize of the figure title
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

def set_parameters_heatmap() -> None:
    FONT_SIZE = 12
    TICK_SIZE = 10
    LEGEND_FONT_SIZE = 12


    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=42)  # fontsize of the figure title
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

def plot_details(df: pd.DataFrame) -> None:
    
    # plt.legend(loc='lower right', framealpha=0.2)

    plt.gca().set_facecolor('#f0f0f0')
    if "example_num" in df:
        l = sorted(df['example_num'].unique())

        processed_l = []
        for ele in l:
            if ele < 100:
                processed_l.append(ele.round(-1))
            else:
                processed_l.append(ele.round(-2))
    elif "epochs" in df:
        l = sorted(df['epochs'].unique())
        processed_l = l
    else:
        raise NotImplementedError
    
    plt.xticks(processed_l, rotation=-55)  # Set rounded x-ticks
    plt.grid(True, which='both', color="white")  # Ensure grid corresponds to ticks

    plt.gca().spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    plt.tight_layout()

NumberFontSize = 15

# For the epochs num, hindsight analysis
FigSize = (5, 8)
# FigSize = (10, 8)

markers = ['o', 's', 'D', '^', '*']
MARKSIZE = 10
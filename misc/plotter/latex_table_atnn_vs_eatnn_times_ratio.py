import numpy as np
import pandas as pd


def times_comparison(df: pd.DataFrame):

    printable_accuracies = __printable_accuracies(df)
    d = __values_to_paste(printable_accuracies)

    caption = """
\\begin{table}[!h]
\\centering
\\footnotesize
\\caption{Wyniki eksperymentów.}
\\begin{tabular}{ll|ccccc}
\\hline
\\multicolumn{2}{c|}{\\textbf{Zbiór danych}} & \\multicolumn{5}{c|}{stosunek metryki $\\frac{\\text{SEATNN}}{\\text{ATNN}}$ [\\%]} \\\\
 & \\tiny{\\makecell{parametr \\\\ szybkości \\\\ dryfu}}  & $\\bar{t}_\\text{warn}$ & $\\bar{n}_\\text{nodes}^{\\text{warn}}$ & $\\bar{n}_\\text{nodes}$ & $T_{\\text{train}}$ & $T_{\\text{overall}}$  \\\\
 
\\hline
"""
    
    
    results = """
\\multirow{1}{*}{\\makecell{electricity}} 
 & & """ + d[0][0] + """ & """ + d[0][1] + """ & """ + d[0][2] + """ & """ + d[0][3] + """ & """ + d[0][4] + """ \\\\


\\multirow{1}{*}{\\makecell{weather}} 
 & & """ + d[1][0] + """ & """ + d[1][1] + """ & """ + d[1][2] + """ & """ + d[1][3] + """ & """ + d[1][4] + """ \\\\


\\multirow{1}{*}{\\makecell{covertype}} 
 & & """ + d[2][0] + """ & """ + d[2][1] + """ & """ + d[2][2] + """ & """ + d[2][3] + """ & """ + d[2][4] + """ \\\\


\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} 
 & & """ + d[3][0] + """ & """ + d[3][1] + """ & """ + d[3][2] + """ & """ + d[3][3] + """ & """ + d[3][4] + """ \\\\


\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} 
 & & """ + d[4][0] + """ & """ + d[4][1] + """ & """ + d[4][2] + """ & """ + d[4][3] + """ & """ + d[4][4] + """ \\\\



\\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$   & """ + d[5][0] + """ & """ + d[5][1] + """ & """ + d[5][2] + """ & """ + d[5][3] + """ & """ + d[5][4] + """ \\\\
 & $x=0,5$   & """ + d[6][0] + """ & """ + d[6][1] + """ & """ + d[6][2] + """ & """ + d[6][3] + """ & """ + d[6][4] + """ \\\\
 & $x=1$     & """ + d[7][0] + """ & """ + d[7][1] + """ & """ + d[7][2] + """ & """ + d[7][3] + """ & """ + d[7][4] + """ \\\\
 & $x=2$     & """ + d[8][0] + """ & """ + d[8][1] + """ & """ + d[8][2] + """ & """ + d[8][3] + """ & """ + d[8][4] + """ \\\\
\\hline 


\\multirow{4}{*}{\\makecell{fashion\\textsubscript{inc}}} 
 & $x=0,1$   & """ + d[9 ][0] + """ & """ + d[9 ][1] + """ & """ + d[9 ][2] + """ & """ + d[9 ][3] + """ & """ + d[9 ][4] + """ \\\\
 & $x=0,5$   & """ + d[10][0] + """ & """ + d[10][1] + """ & """ + d[10][2] + """ & """ + d[10][3] + """ & """ + d[10][4] + """ \\\\
 & $x=1$     & """ + d[11][0] + """ & """ + d[11][1] + """ & """ + d[11][2] + """ & """ + d[11][3] + """ & """ + d[11][4] + """ \\\\
 & $x=2$     & """ + d[12][0] + """ & """ + d[12][1] + """ & """ + d[12][2] + """ & """ + d[12][3] + """ & """ + d[12][4] + """ \\\\
\\hline 

"""
    
    
    end = """
\\end{tabular}
\\label{tab:seatnn_vs_atnn_time_ratios}
\\end{table}
    """

    
    
    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    for dataset in df["Dataset"].unique():
        results[dataset] = {}
        temp_atnn = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn")]
        temp_eatnn = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2")]

        for metric in ["Warn samples", "Warn nodes", "Overall nodes", "Time Train", "Time Overall"]:
            atnn_value = float(temp_atnn[metric].values[0])
            eatnn_value = float(temp_eatnn[metric].values[0])

            ratio = eatnn_value / atnn_value

            printable_ratio = "{:.2f}".format(ratio).replace(".", ",")
            if ratio < 1:
                printable_ratio = f"\\textbf{{{printable_ratio}}}"

            results[dataset][metric] = printable_ratio

    return results

def __values_to_paste(data: dict) -> list[list]:
    datasets = [
        "elec",
        "weather_norm",
        "covtype_norm",
        "sea_abr",
        # "sea_inc",
        "mnist_abrupt_atnn_like",
        "mnist_inc_20k_0.1x",
        "mnist_inc_20k_0.5x",
        "mnist_inc_20k_1x",
        "mnist_inc_20k_2x",
        "fashion_inc_20k_0.1x",
        "fashion_inc_20k_0.5x",
        "fashion_inc_20k_1x",
        "fashion_inc_20k_2x",
    ]

    results = []
    for dataset in datasets:
        results.append(__get_row(data, dataset))

    return results

def __get_row(data: dict, dataset: str) -> list:
    return [
        data[dataset]["Warn samples"],
        data[dataset]["Warn nodes"],
        data[dataset]["Overall nodes"],
        data[dataset]["Time Train"],
        data[dataset]["Time Overall"],
        data[dataset]["Warn samples"],
        data[dataset]["Warn nodes"],
        data[dataset]["Overall nodes"],
        data[dataset]["Time Train"],
        data[dataset]["Time Overall"],
    ]
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
\\begin{tabular}{ll|ccccc|ccccc}
\\hline
\\multicolumn{2}{c|}{\\textbf{Zbiór danych}} & \\multicolumn{5}{c|}{ATNN} & \\multicolumn{5}{c}{SEATNN} \\\\
 & \\tiny{\\makecell{parametr \\\\ szybkości \\\\ dryfu}}  & $d_\\text{warn}$ & $n_\\text{nodes}^{\\text{warn}}$ & $n_\\text{nodes}$ & $T_{\\text{train}}$ & $T_{\\text{overall}}$ & $d_\\text{warn}$ & $n_\\text{nodes}^{\\text{warn}}$ & $n_\\text{nodes}$ & $T_{\\text{train}}$ & $T_{\\text{overall}}$ \\\\

  & & $\\%$ & & & [s] & [s] & $\\%$ & & & [s] & [s] \\\\
 
\\hline
    """
    
    
    results = """
\\multirow{1}{*}{\\makecell{electricity}} 
 & & """ + d[0][0] + """ & """ + d[0][1] + """ & """ + d[0][2] + """ & """ + d[0][3] + """ & """ + d[0][4] + """ & """ + d[0][5] + """ & """ + d[0][6] + """ & """ + d[0][7] + """ & """ + d[0][8] + """ & """ + d[0][9] + """ \\\\


\\multirow{1}{*}{\\makecell{weather}} 
 & & """ + d[1][0] + """ & """ + d[1][1] + """ & """ + d[1][2] + """ & """ + d[1][3] + """ & """ + d[1][4] + """ & """ + d[1][5] + """ & """ + d[1][6] + """ & """ + d[1][7] + """ & """ + d[1][8] + """ & """ + d[1][9] + """ \\\\


\\multirow{1}{*}{\\makecell{covertype}} 
 & & """ + d[2][0] + """ & """ + d[2][1] + """ & """ + d[2][2] + """ & """ + d[2][3] + """ & """ + d[2][4] + """ & """ + d[2][5] + """ & """ + d[2][6] + """ & """ + d[2][7] + """ & """ + d[2][8] + """ & """ + d[2][9] + """ \\\\


\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} 
 & & """ + d[3][0] + """ & """ + d[3][1] + """ & """ + d[3][2] + """ & """ + d[3][3] + """ & """ + d[3][4] + """ & """ + d[3][5] + """ & """ + d[3][6] + """ & """ + d[3][7] + """ & """ + d[3][8] + """ & """ + d[3][9] + """ \\\\


\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} 
 & & """ + d[4][0] + """ & """ + d[4][1] + """ & """ + d[4][2] + """ & """ + d[4][3] + """ & """ + d[4][4] + """ & """ + d[4][5] + """ & """ + d[4][6] + """ & """ + d[4][7] + """ & """ + d[4][8] + """ & """ + d[4][9] + """ \\\\




\\multirow{1}{*}{\\makecell{SEA\\textsubscript{inc}}} 
 & & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- \\\\


\\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$   & """ + d[5][0] + """ & """ + d[5][1] + """ & """ + d[5][2] + """ & """ + d[5][3] + """ & """ + d[5][4] + """ & """ + d[5][5] + """ & """ + d[5][6] + """ & """ + d[5][7] + """ & """ + d[5][8] + """ & """ + d[5][9] + """ \\\\
 & $x=0,5$   & """ + d[6][0] + """ & """ + d[6][1] + """ & """ + d[6][2] + """ & """ + d[6][3] + """ & """ + d[6][4] + """ & """ + d[6][5] + """ & """ + d[6][6] + """ & """ + d[6][7] + """ & """ + d[6][8] + """ & """ + d[6][9] + """ \\\\
 & $x=1$     & """ + d[7][0] + """ & """ + d[7][1] + """ & """ + d[7][2] + """ & """ + d[7][3] + """ & """ + d[7][4] + """ & """ + d[7][5] + """ & """ + d[7][6] + """ & """ + d[7][7] + """ & """ + d[7][8] + """ & """ + d[7][9] + """ \\\\
 & $x=2$     & """ + d[8][0] + """ & """ + d[8][1] + """ & """ + d[8][2] + """ & """ + d[8][3] + """ & """ + d[8][4] + """ & """ + d[8][5] + """ & """ + d[8][6] + """ & """ + d[8][7] + """ & """ + d[8][8] + """ & """ + d[8][9] + """ \\\\
\\hline 


\\multirow{4}{*}{\\makecell{fashion\\textsubscript{inc}}} 
 & $x=0,1$   & """ + d[9 ][0] + """ & """ + d[9 ][1] + """ & """ + d[9 ][2] + """ & """ + d[9 ][3] + """ & """ + d[9 ][4] + """ & """ + d[9 ][5] + """ & """ + d[9 ][6] + """ & """ + d[9 ][7] + """ & """ + d[9 ][8] + """ & """ + d[9 ][9] + """ \\\\
 & $x=0,5$   & """ + d[10][0] + """ & """ + d[10][1] + """ & """ + d[10][2] + """ & """ + d[10][3] + """ & """ + d[10][4] + """ & """ + d[10][5] + """ & """ + d[10][6] + """ & """ + d[10][7] + """ & """ + d[10][8] + """ & """ + d[10][9] + """ \\\\
 & $x=1$     & """ + d[11][0] + """ & """ + d[11][1] + """ & """ + d[11][2] + """ & """ + d[11][3] + """ & """ + d[11][4] + """ & """ + d[11][5] + """ & """ + d[11][6] + """ & """ + d[11][7] + """ & """ + d[11][8] + """ & """ + d[11][9] + """ \\\\
 & $x=2$     & """ + d[12][0] + """ & """ + d[12][1] + """ & """ + d[12][2] + """ & """ + d[12][3] + """ & """ + d[12][4] + """ & """ + d[12][5] + """ & """ + d[12][6] + """ & """ + d[12][7] + """ & """ + d[12][8] + """ & """ + d[12][9] + """ \\\\
\\hline 

    """
    
    
    end = """
\\end{tabular}
\\label{tab:seatnn_vs_atnn_times}
\\end{table}
    """

    
    
    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    for dataset in df["Dataset"].unique():
        results[dataset] = {}
        results[dataset]["atnn"] = {}
        results[dataset]["eatnn2"] = {}
        temp_atnn = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn")]
        temp_eatnn = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2")]

        for metric in ["Warn samples", "Warn nodes", "Overall nodes", "Time Train", "Time Overall"]:
            atnn_value = float(temp_atnn[metric].values[0])
            eatnn_value = float(temp_eatnn[metric].values[0])

            printable_atnn_value = "{:.2f}".format(atnn_value).replace(".", ",")
            if atnn_value <= eatnn_value:
                printable_atnn_value = f"\\textbf{{{printable_atnn_value}}}"

            printable_eatnn_value = "{:.2f}".format(eatnn_value).replace(".", ",")
            if eatnn_value <= atnn_value:
                printable_eatnn_value = f"\\textbf{{{printable_eatnn_value}}}"

            results[dataset]["atnn"][metric] = printable_atnn_value
            results[dataset]["eatnn2"][metric] = printable_eatnn_value

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
        data[dataset]["atnn"]["Warn samples"],
        data[dataset]["atnn"]["Warn nodes"],
        data[dataset]["atnn"]["Overall nodes"],
        data[dataset]["atnn"]["Time Train"],
        data[dataset]["atnn"]["Time Overall"],
        data[dataset]["eatnn2"]["Warn samples"],
        data[dataset]["eatnn2"]["Warn nodes"],
        data[dataset]["eatnn2"]["Overall nodes"],
        data[dataset]["eatnn2"]["Time Train"],
        data[dataset]["eatnn2"]["Time Overall"],
    ]
import numpy as np
import pandas as pd


def cand_accuracies_table(df: pd.DataFrame):
    d = __printable_accuracies(df)

    caption = """
\\begin{table}[!h]
\\centering
\\caption{Wyniki eksperymentów dla różnych wariantów algorytmu CAND. Tabela przedstawia dokładności. Pogrubioną czcionką zaznaczono najlepszy wynik uzyskany na danym zbiorze danych.}
% p10, p30, m6p30, m8p30, m10p30, najlepszy z sub1, sub2, sub3
\\begin{tabular}{ll|ccccc}
\\hline
\\multicolumn{2}{c|}{\\textbf{Zbiór danych}} & \\multicolumn{5}{c}{Dokładność [\\%]} \\\\
 & & \\multicolumn{2}{c}{CAND} & \\multicolumn{3}{c}{CAND\\textsubscript{sub}} \\\\
 
 & \\footnotesize{\\makecell{parametr \\\\ szybkości \\\\ dryfu}} 
 & $|P|=10$ & $|P|=30$ & \\makecell{$|M|=6$ \\\\ $|P|=30$} & \\makecell{$|M|=8$ \\\\ $|P|=30$} & \\makecell{$|M|=10$ \\\\ $|P|=30$}  \\\\
 
\\hline
    """

    end = """
\\hline 

\\end{tabular}
\\label{tab:cand_accuracies}
\\end{table}
"""

    results = """
\\multirow{1}{*}{\\makecell{electricity}} & & """ + d["elec"][1] + """ & """ + d["elec"][2] + """ & """ + \
              d["elec"][3] + """ & """ + \
              d["elec"][4] + """ & """ + \
              d["elec"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & & """ + d["weather_norm"][1] + """ & """ + d["weather_norm"][2] + """ & """ + \
              d["weather_norm"][3] + """ & """ + \
              d["weather_norm"][4] + """ & """ + \
              d["weather_norm"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{covertype}} & & """ + d["covtype_norm"][1] + """ & """ + d["covtype_norm"][2] + """ & """ + \
              d["covtype_norm"][3] + """ & """ + \
              d["covtype_norm"][4] + """ & """ + \
              d["covtype_norm"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} & & """ + d["sea_abr"][1] + """ & """ + d["sea_abr"][2] + """ & """ + \
              d["sea_abr"][3] + """ & """ + \
              d["sea_abr"][4] + """ & """ + \
              d["sea_abr"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{inc}}} & & """ + d["sea_inc"][1] + """ & """ + d["sea_inc"][2] + """ & """ + \
              d["sea_inc"][3] + """ & """ + \
              d["sea_inc"][4] + """ & """ + \
              d["sea_inc"][5] + """ \\\\


\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & & """ + d["mnist_abrupt_atnn_like"][1] + """ & """ + d["mnist_abrupt_atnn_like"][2] + """ & """ + d["mnist_abrupt_atnn_like"][3] + """ & """ + \
              d["mnist_abrupt_atnn_like"][4] + """ & """ + \
              d["mnist_abrupt_atnn_like"][5] + """ \\\\

\\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$ & """ + d["mnist_inc_20k_0.1x"][1] + """ & """ + d["mnist_inc_20k_0.1x"][2] + """ & """ + \
              d["mnist_inc_20k_0.1x"][3]+ """ & """ + \
              d["mnist_inc_20k_0.1x"][4] + """ & """ + \
              d["mnist_inc_20k_0.1x"][5] + """ \\\\
 & $x=0,5$  & """ + d["mnist_inc_20k_0.5x"][1] + """ & """ + d["mnist_inc_20k_0.5x"][2] + """ & """ + \
              d["mnist_inc_20k_0.5x"][3]+ """ & """ + \
              d["mnist_inc_20k_0.5x"][4] + """ & """ + \
              d["mnist_inc_20k_0.5x"][5] + """ \\\\
 & $x=1$    & """ + d["mnist_inc_20k_1x"][1] + """ & """ + d["mnist_inc_20k_1x"][2] + """ & """ + \
              d["mnist_inc_20k_1x"][3]+ """ & """ + \
              d["mnist_inc_20k_1x"][4] + """ & """ + \
              d["mnist_inc_20k_1x"][5] + """ \\\\
 & $x=2$    & """ + d["mnist_inc_20k_2x"][1] + """ & """ + d["mnist_inc_20k_2x"][2] + """ & """ + \
              d["mnist_inc_20k_2x"][3]+ """ & """ + \
              d["mnist_inc_20k_2x"][4] + """ & """ + \
              d["mnist_inc_20k_2x"][5] + """ \\\\
    """

    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    printables = {}

    keys_to_sizes = {
        1: "P10_M10",
        2: "P30_M30",
        3: "P30_M06",
        4: "P30_M08",
        5: "P30_M10"
    }
    for dataset in df["Dataset"].unique():
        vals = {}
        results[dataset] = {}
        results[dataset]["cand"] = {}
        results[dataset]["cand"]["P10_M10"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize10_Msize10_Bpth0.0")]
        results[dataset]["cand"]["P30_M30"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize30_Bpth0.0")]
        results[dataset]["cand"]["P30_M06"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize6_Bpth0.0")]
        results[dataset]["cand"]["P30_M08"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize8_Bpth0.0")]
        results[dataset]["cand"]["P30_M10"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize10_Bpth0.0")]

        vals[dataset] = {}
        vals[dataset]["P10_M10"] = float(results[dataset]["cand"]["P10_M10"]["Accuracy"].values[0])
        vals[dataset]["P30_M30"] = float(results[dataset]["cand"]["P30_M30"]["Accuracy"].values[0])
        vals[dataset]["P30_M06"] = float(results[dataset]["cand"]["P30_M06"]["Accuracy"].values[0])
        vals[dataset]["P30_M08"] = float(results[dataset]["cand"]["P30_M08"]["Accuracy"].values[0])
        vals[dataset]["P30_M10"] = float(results[dataset]["cand"]["P30_M10"]["Accuracy"].values[0])


        printables[dataset] = {}
        printables[dataset][1] = "{:.2f}".format(vals[dataset]["P10_M10"]).replace(".", ",")
        printables[dataset][2] = "{:.2f}".format(vals[dataset]["P30_M30"]).replace(".", ",")
        printables[dataset][3] = "{:.2f}".format(vals[dataset]["P30_M06"]).replace(".", ",")
        printables[dataset][4] = "{:.2f}".format(vals[dataset]["P30_M08"]).replace(".", ",")
        printables[dataset][5] = "{:.2f}".format(vals[dataset]["P30_M10"]).replace(".", ",")


        for key in printables[dataset].keys():
            if vals[dataset][keys_to_sizes[key]] == max(vals[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

    return printables


import numpy as np
import pandas as pd


def atnn_eatnn_gammas_accuracies_table(df: pd.DataFrame):
    printable_accuracies = __printable_accuracies(df)

    caption = """
\\begin{table}[!h]
\\centering
\\caption{Wyniki eksperymentów porównujących dokładności algorytm SEATNN dla różnych wartości parametru $gamma$. Pogrubioną czcionką zaznaczono większą dokładność dla danego zbioru danych.}
\\begin{tabular}{llcccc}
\\hline
\\multicolumn{2}{c}{\\textbf{Zbiór danych}} & \\multicolumn{4}{c}{\\textbf{Dokładność}} \\\\ 
 &  & ATNN & \\multicolumn{3}{c}{SEATNN} \\\\
 & \\scriptsize{\\makecell{parametr \\\\ szybkości dryfu}}  & & $\gamma=1$ & $\gamma=0.5$ & $\gamma=0.1$ \\\\
\\hline
    """

    end = """
\\hline
\\end{tabular}
\\label{tab:atnn_eatnn_comparison_accuracies}
\\end{table}
    """

    results = """
\\multirow{1}{*}{\\makecell{electricity}} & & """ + printable_accuracies["elec"].get("atnn", "-") + """ & """ + printable_accuracies["elec"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["elec"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["elec"]["eatnn"].get("0.1", "-") + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & & """ + printable_accuracies["weather_norm"].get("atnn", "-") + """ & """ + printable_accuracies["weather_norm"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["weather_norm"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["weather_norm"]["eatnn"].get("0.1", "-") + """ \\\\

\\multirow{1}{*}{\\makecell{covertype}} & & """ + printable_accuracies["covtype_norm"].get("atnn", "-") + """ & """ + printable_accuracies["covtype_norm"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["covtype_norm"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["covtype_norm"]["eatnn"].get("0.1", "-") + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} & & """ + printable_accuracies["sea_abr"].get("atnn", "-") + """ & """ + printable_accuracies["sea_abr"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["sea_abr"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["sea_abr"]["eatnn"].get("0.1", "-") + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{inc}}} & & - & - & - & - \\\\


\\multirow{1}{*}{\\makecell{MNIST}} & & & & & \\\\

\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & & """ + printable_accuracies["mnist_abrupt_atnn_like"].get("atnn", "-") + """ & """ + printable_accuracies["mnist_abrupt_atnn_like"]["eatnn"][
                  "1.0"] + """ & """ + printable_accuracies["mnist_abrupt_atnn_like"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["mnist_abrupt_atnn_like"]["eatnn"].get("0.1", "-") + """ \\\\

\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$ & """ + printable_accuracies["mnist_inc_20k_0.1x"].get("atnn", "-") + """ & """ + printable_accuracies["mnist_inc_20k_0.1x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["mnist_inc_20k_0.1x"]["eatnn"].get("0.5", "-")+ """ & """ + \
              printable_accuracies["mnist_inc_20k_0.1x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=0,5$  & """ + printable_accuracies["mnist_inc_20k_0.5x"].get("atnn", "-") + """ & """ + printable_accuracies["mnist_inc_20k_0.5x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["mnist_inc_20k_0.5x"]["eatnn"].get("0.5", "-")+ """ & """ + \
              printable_accuracies["mnist_inc_20k_0.5x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=1$    & """ + printable_accuracies["mnist_inc_20k_1x"].get("atnn", "-") + """ & """ + printable_accuracies["mnist_inc_20k_1x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["mnist_inc_20k_1x"]["eatnn"].get("0.5", "-")+ """ & """ + \
              printable_accuracies["mnist_inc_20k_1x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=2$    & """ + printable_accuracies["mnist_inc_20k_2x"].get("atnn", "-") + """ & """ + printable_accuracies["mnist_inc_20k_2x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["mnist_inc_20k_2x"]["eatnn"].get("0.5", "-")+ """ & """ + \
              printable_accuracies["mnist_inc_20k_2x"]["eatnn"].get("0.1", "-") + """ \\\\

\\multirow{1}{*}{\\makecell{fashion}} & & & & \\\\

\\multirow{4}{*}{\\makecell{fashion\\textsubscript{inc}}} 
 & $x=0,1$ & """ + printable_accuracies["fashion_inc_40k_0.1x"].get("atnn", "-") + """ & """ + printable_accuracies["fashion_inc_40k_0.1x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_0.1x"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_0.1x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=0,5$  & """ + printable_accuracies["fashion_inc_40k_0.5x"].get("atnn", "-") + """ & """ + printable_accuracies["fashion_inc_40k_0.5x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_0.5x"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_0.5x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=1$    & """ + printable_accuracies["fashion_inc_40k_1x"].get("atnn", "-") + """ & """ + printable_accuracies["fashion_inc_40k_1x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_1x"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_1x"]["eatnn"].get("0.1", "-") + """ \\\\
 & $x=2$    & """ + printable_accuracies["fashion_inc_40k_2x"].get("atnn", "-") + """ & """ + printable_accuracies["fashion_inc_40k_2x"]["eatnn"].get("1.0", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_2x"]["eatnn"].get("0.5", "-") + """ & """ + \
              printable_accuracies["fashion_inc_40k_2x"]["eatnn"].get("0.1", "-") + """ \\\\

    """

    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    for dataset in df["Dataset"].unique():
        results[dataset] = {}
        results[dataset]["eatnn"] = {}
        results[dataset]["eatnn"]["1.0"] = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma1.0")]
        results[dataset]["eatnn"]["0.5"] = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma0.5")]
        results[dataset]["eatnn"]["0.1"] = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma0.1")]
        results[dataset]["atnn"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn")]

        try:
            atnn_acc = float(results[dataset]["atnn"]["Accuracy"].values[0])
        except Exception:
            atnn_acc = 0.0
        try:
            acc10 = float(results[dataset]["eatnn"]["1.0"]["Accuracy"].values[0])
        except Exception:
            acc10 = 0.0
        try:
            acc05 = float(results[dataset]["eatnn"]["0.5"]["Accuracy"].values[0])
        except Exception:
            acc05 = 0.0
        try:
            acc01 = float(results[dataset]["eatnn"]["0.1"]["Accuracy"].values[0])
        except Exception:
            acc01 = 0.0

        printable_atnn_acc = "{:.2f}".format(atnn_acc).replace(".", ",")
        if atnn_acc >= acc10 and atnn_acc >= acc05 and atnn_acc >= acc01:
            printable_atnn_acc = f"\\textbf{{{printable_atnn_acc}}}"

        printable_acc10 = "{:.2f}".format(acc10).replace(".", ",")
        if acc10 >= atnn_acc and acc10 >= acc05 and acc10 >= acc01:
            printable_acc10 = f"\\textbf{{{printable_acc10}}}"

        printable_acc05 = "{:.2f}".format(acc05).replace(".", ",")
        if acc05 >= atnn_acc and acc05 >= acc10 and acc05 >= acc01:
            printable_acc05 = f"\\textbf{{{printable_acc05}}}"

        printable_acc01 = "{:.2f}".format(acc01).replace(".", ",")
        if acc01 >= atnn_acc and acc01 >= acc10 and acc01 >= acc05:
            printable_acc01 = f"\\textbf{{{printable_acc01}}}"

        results[dataset]["atnn"] = printable_atnn_acc
        results[dataset]["eatnn"]["1.0"] = printable_acc10
        results[dataset]["eatnn"]["0.5"] = printable_acc05
        results[dataset]["eatnn"]["0.1"] = printable_acc01

    return results


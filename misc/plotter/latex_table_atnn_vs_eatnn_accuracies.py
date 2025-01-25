import numpy as np
import pandas as pd


def accuracy_comparison(df: pd.DataFrame):
    printable_accuracies = __printable_accuracies(df)
    
    caption = """
\\begin{table}[!h]
\\centering
\\caption{Wyniki eksperymentów porównujących dokładności algorytmów ATNN i SEATNN. Pogrubioną czcionką zaznaczono większą dokładność dla danego zbioru danych.}
\\begin{tabular}{llcc}
\\hline
\\multicolumn{2}{c}{\\textbf{Zbiór danych}} & \\multicolumn{2}{c}{\\textbf{Dokładność}} \\\\ 
 & \\scriptsize{\\makecell{parametr \\\\ szybkości dryfu}}  & ATNN & SEATNN \\\\
\\hline
    """
    
    end = """
\\hline
\\end{tabular}
\\label{tab:atnn_eatnn_comparison_accuracies}
\\end{table}
    """
    
    results = """
\\multirow{1}{*}{\\makecell{electricity}} & & """ + printable_accuracies["elec"]["atnn"] + """ & """ + printable_accuracies["elec"]["eatnn2"] + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & & """ + printable_accuracies["weather_norm"]["atnn"] + """ & """ + printable_accuracies["weather_norm"]["eatnn2"] + """ \\\\

\\multirow{1}{*}{\\makecell{covertype}} & & """ + printable_accuracies["covtype_norm"]["atnn"] + """ & """ + printable_accuracies["covtype_norm"]["eatnn2"] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} & & """ + printable_accuracies["sea_abr"]["atnn"] + """ & """ + printable_accuracies["sea_abr"]["eatnn2"] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{inc}}} & & --- & --- \\\\

\\multirow{1}{*}{\\makecell{MNIST}} & & & \\\\

\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & & """ + printable_accuracies["mnist_abrupt_atnn_like"]["atnn"] + """ & """ + printable_accuracies["mnist_abrupt_atnn_like"]["eatnn2"] + """ \\\\

\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$ & """ + printable_accuracies["mnist_inc_20k_0.1x"]["atnn"] + """ & """ + printable_accuracies["mnist_inc_20k_0.1x"]["eatnn2"] + """ \\\\
 & $x=0,5$  & """ + printable_accuracies["mnist_inc_20k_0.5x"]["atnn"] + """ & """ + printable_accuracies["mnist_inc_20k_0.5x"]["eatnn2"] + """ \\\\
 & $x=1$    & """ + printable_accuracies["mnist_inc_20k_1x"]["atnn"] + """ & """ + printable_accuracies["mnist_inc_20k_1x"]["eatnn2"] + """ \\\\
 & $x=2$    & """ + printable_accuracies["mnist_inc_20k_2x"]["atnn"] + """ & """ + printable_accuracies["mnist_inc_20k_2x"]["eatnn2"] + """ \\\\

\\multirow{1}{*}{\\makecell{fashion}} & & & \\\\

\\multirow{4}{*}{\\makecell{fashion\\textsubscript{inc}}} 
 & $x=0,1$ & """ + printable_accuracies["fashion_inc_20k_0.1x"]["atnn"] + """ & """ + printable_accuracies["fashion_inc_20k_0.1x"]["eatnn2"] + """ \\\\
 & $x=0,5$  & """ + printable_accuracies["fashion_inc_20k_0.5x"]["atnn"] + """ & """ + printable_accuracies["fashion_inc_20k_0.5x"]["eatnn2"] + """ \\\\
 & $x=1$    & """ + printable_accuracies["fashion_inc_20k_1x"]["atnn"] + """ & """ + printable_accuracies["fashion_inc_20k_1x"]["eatnn2"] + """ \\\\
 & $x=2$    & """ + printable_accuracies["fashion_inc_20k_2x"]["atnn"] + """ & """ + printable_accuracies["fashion_inc_20k_2x"]["eatnn2"] + """ \\\\

    """
    
    
    text = caption + results + end
    return text

def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    for dataset in df["Dataset"].unique():
        results[dataset] = {}
        results[dataset]["atnn"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn")]
        results[dataset]["eatnn2"] = df[(df["Dataset"] == dataset) & (df["Method"] == "eatnn2")]

        atnn_acc = float(results[dataset]["atnn"]["Accuracy"].values[0])
        eatnn_acc = float(results[dataset]["eatnn2"]["Accuracy"].values[0])

        printable_atnn_accuracy = "{:.2f}".format(atnn_acc).replace(".", ",")
        if atnn_acc >= eatnn_acc:
            printable_atnn_accuracy = f"\\textbf{{{printable_atnn_accuracy}}}"

        printable_eatnn_accuracy = "{:.2f}".format(eatnn_acc).replace(".", ",")
        if eatnn_acc >= atnn_acc:
            printable_eatnn_accuracy = f"\\textbf{{{printable_eatnn_accuracy}}}"

        results[dataset]["atnn"] = printable_atnn_accuracy
        results[dataset]["eatnn2"] = printable_eatnn_accuracy

    return results


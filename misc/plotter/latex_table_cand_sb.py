import numpy as np
import pandas as pd


def cand_sb_table(df: pd.DataFrame):
    d = __printable_accuracies(df)

    caption = """
\\begin{table}[H]
\\centering
\\caption{Wyniki eksperymentów dla różnych wariantów algorytmu CAND\\textsubscript{sub}\\textsuperscript{SB}. Tabela przedstawia dokładności oraz czasy wykonania algorytmu dla różnych wartości parametru $sb$. Eksperymenty wykonano dla wariantu $|P|=30$, $|M|=10$}
\\begin{tabular}{ll|cccc|cccc}
\\hline
\\multicolumn{2}{c|}{\\textbf{Zbiór danych}} & \\multicolumn{4}{c|}{Dokładność [\\%]} & \\multicolumn{4}{c}{Czas wykonania [s]} \\\\

 & \\multirow{2}{*}{\\scriptsize{\\makecell{parametr \\\\ szybkości \\\\ dryfu}}}  & \\multicolumn{4}{c|}{$sb=$} & \\multicolumn{4}{c}{$sb=$} \\\\ 
 & 
 & $0$ & $0,3$ & $0,6$ & $1,0$ & $0$ & $0,3$ & $0,6$ & $1,0$ \\\\
 
\\hline
    """

    end = """
\\hline 

\\end{tabular}
\\label{tab:cand_sb}
\\end{table}
"""

    results = ("""
\\multirow{1}{*}{\\makecell{electricity}} & & """ + \
              d["elec"][1] + """ & """ + \
              d["elec"][2] + """ & """ + \
              d["elec"][3] + """ & """ + \
              d["elec"][4] + """ & """ + \
              d["elec"][5] + """ & """ + \
              d["elec"][6] + """ & """ + \
              d["elec"][7] + """ & """ + \
              d["elec"][8] + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & & """ + \
              d["weather_norm"][1] + """ & """ + \
              d["weather_norm"][2] + """ & """ + \
              d["weather_norm"][3] + """ & """ + \
              d["weather_norm"][4] + """ & """ + \
              d["weather_norm"][5] + """ & """ + \
              d["weather_norm"][6] + """ & """ + \
              d["weather_norm"][7] + """ & """ + \
              d["weather_norm"][8] + """ \\\\

\\multirow{1}{*}{\\makecell{covertype}} & & """ + \
              d["covtype_norm"][1] + """ & """ + \
              d["covtype_norm"][2] + """ & """ + \
              d["covtype_norm"][3] + """ & """ + \
              d["covtype_norm"][4] + """ & """ + \
              d["covtype_norm"][5] + """ & """ + \
              d["covtype_norm"][6] + """ & """ + \
              d["covtype_norm"][7] + """ & """ + \
              d["covtype_norm"][8] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{abr}}} & & """ + \
              d["sea_abr"][1] + """ & """ + \
              d["sea_abr"][2] + """ & """ + \
              d["sea_abr"][3] + """ & """ + \
              d["sea_abr"][4] + """ & """ + \
              d["sea_abr"][5] + """ & """ + \
              d["sea_abr"][6] + """ & """ + \
              d["sea_abr"][7] + """ & """ + \
              d["sea_abr"][8] + """ \\\\

\\multirow{1}{*}{\\makecell{SEA\\textsubscript{inc}}} & & """ + \
              d["sea_inc"][1] + """ & """ + \
              d["sea_inc"][2] + """ & """ + \
              d["sea_inc"][3] + """ & """ + \
              d["sea_inc"][4] + """ & """ + \
              d["sea_inc"][5] + """ & """ + \
              d["sea_inc"][6] + """ & """ + \
              d["sea_inc"][7] + """ & """ + \
              d["sea_inc"][8] + """ \\\\

\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & & """ + d["mnist_abrupt_atnn_like"][1] + """ & """ + d["mnist_abrupt_atnn_like"][2] + """ & """ + d["mnist_abrupt_atnn_like"][3] + """ & """ + \
              d["mnist_abrupt_atnn_like"][4] + """ & """ + \
              d["mnist_abrupt_atnn_like"][5] + """ & """ +
               d["mnist_abrupt_atnn_like"][6] + """ & """ +
               d["mnist_abrupt_atnn_like"][7] + """ & """ + \
               d["mnist_abrupt_atnn_like"][8] + """ \\\\

\\multirow{4}{*}{\\makecell{MNIST\\textsubscript{inc}}} 
 & $x=0,1$ & """ + d["mnist_inc_20k_0.1x"][1] + """ & """ + d["mnist_inc_20k_0.1x"][2] + """ & """ + \
              d["mnist_inc_20k_0.1x"][3]+ """ & """ + \
              d["mnist_inc_20k_0.1x"][4] + """ & """ + \
               d["mnist_inc_20k_0.1x"][5] + """ & """ +
               d["mnist_inc_20k_0.1x"][6] + """ & """ +
               d["mnist_inc_20k_0.1x"][7] + """ & """ + \
               d["mnist_inc_20k_0.1x"][8] + """ \\\\
 & $x=0,5$  & """ + d["mnist_inc_20k_0.5x"][1] + """ & """ + d["mnist_inc_20k_0.5x"][2] + """ & """ + \
              d["mnist_inc_20k_0.5x"][3]+ """ & """ + \
              d["mnist_inc_20k_0.5x"][4] + """ & """ + \
               d["mnist_inc_20k_0.5x"][5] + """ & """ +
               d["mnist_inc_20k_0.5x"][6] + """ & """ +
               d["mnist_inc_20k_0.5x"][7] + """ & """ + \
               d["mnist_inc_20k_0.5x"][8] + """ \\\\
 & $x=1$    & """ + d["mnist_inc_20k_1x"][1] + """ & """ + d["mnist_inc_20k_1x"][2] + """ & """ + \
              d["mnist_inc_20k_1x"][3]+ """ & """ + \
              d["mnist_inc_20k_1x"][4] + """ & """ + \
               d["mnist_inc_20k_1x"][5] + """ & """ +
               d["mnist_inc_20k_1x"][6] + """ & """ +
               d["mnist_inc_20k_1x"][7] + """ & """ +
               d["mnist_inc_20k_1x"][8] + """ \\\\
 & $x=2$    & """ + d["mnist_inc_20k_2x"][1] + """ & """ + d["mnist_inc_20k_2x"][2] + """ & """ + \
              d["mnist_inc_20k_2x"][3]+ """ & """ + \
              d["mnist_inc_20k_2x"][4] + """ & """ + \
               d["mnist_inc_20k_2x"][5] + """ & """ +
               d["mnist_inc_20k_2x"][6] + """ & """ +
               d["mnist_inc_20k_2x"][7] + """ & """ +
               d["mnist_inc_20k_2x"][8] + """ \\\\
    """)

    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    printables = {}
    keys_to_sb = {
        1: "0.0",
        2: "0.2",
        3: "0.6",
        4: "1.0",
        5: "0.0",
        6: "0.2",
        7: "0.6",
        8: "1.0"
    }

    for dataset in df["Dataset"].unique():
        accs = {}
        times = {}
        results[dataset] = {}
        results[dataset]["cand"] = {}
        results[dataset]["cand"]["0.0"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize10_Bpth0.0")]
        results[dataset]["cand"]["0.2"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize10_Bpth0.2")]
        results[dataset]["cand"]["0.6"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize10_Bpth0.6")]
        results[dataset]["cand"]["1.0"] = df[(df["Dataset"] == dataset) & (df["Method"] == "cand") & (df["Params"] == "Psize30_Msize10_Bpth1.0")]

        print(f"dataset: {dataset}")
        accs[dataset] = {}
        accs[dataset]["0.0"] = float(results[dataset]["cand"]["0.0"]["Accuracy"].values[0])
        accs[dataset]["0.2"] = float(results[dataset]["cand"]["0.2"]["Accuracy"].values[0])
        accs[dataset]["0.6"] = float(results[dataset]["cand"]["0.6"]["Accuracy"].values[0])
        accs[dataset]["1.0"] = float(results[dataset]["cand"]["1.0"]["Accuracy"].values[0])

        times[dataset] = {}
        times[dataset]["0.0"] = float(results[dataset]["cand"]["0.0"]["Time Overall"].values[0])
        times[dataset]["0.2"] = float(results[dataset]["cand"]["0.2"]["Time Overall"].values[0])
        times[dataset]["0.6"] = float(results[dataset]["cand"]["0.6"]["Time Overall"].values[0])
        times[dataset]["1.0"] = float(results[dataset]["cand"]["1.0"]["Time Overall"].values[0])

        printables[dataset] = {}
        printables[dataset][1] = "{:.2f}".format(accs[dataset]["0.0"]).replace(".", ",")
        printables[dataset][2] = "{:.2f}".format(accs[dataset]["0.2"]).replace(".", ",")
        printables[dataset][3] = "{:.2f}".format(accs[dataset]["0.6"]).replace(".", ",")
        printables[dataset][4] = "{:.2f}".format(accs[dataset]["1.0"]).replace(".", ",")

        printables[dataset][5] = "{:.2f}".format(times[dataset]["0.0"]).replace(".", ",")
        printables[dataset][6] = "{:.2f}".format(times[dataset]["0.2"]).replace(".", ",")
        printables[dataset][7] = "{:.2f}".format(times[dataset]["0.6"]).replace(".", ",")
        printables[dataset][8] = "{:.2f}".format(times[dataset]["1.0"]).replace(".", ",")

        for key in [1, 2, 3, 4]:
            if accs[dataset][keys_to_sb[key]] == max(accs[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

        for key in [5, 6, 7, 8]:
            if times[dataset][keys_to_sb[key]] == min(times[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

    return printables


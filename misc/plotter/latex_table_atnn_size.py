import numpy as np
import pandas as pd


def atnn_size_accuracy_comparison_table(df: pd.DataFrame):
    d = __printable_accuracies(df)

    caption = """
\\begin{table}[H]
\\centering
\\caption{Dokładność ze względu na rozmiar warstwy ukrytej. Podczas eksperymentów szybkość uczenia ustawiono na wartość $\\eta=0,02$, a parametr regularyzacji $\\lambda=5000$.}
\\begin{tabular}{llccccc}
\\hline
\\multicolumn{1}{c}{\\textbf{Zbiór danych}} & \multicolumn{5}{c}{\\textbf{Dokładność}} \\\\ 
 &  \\multicolumn{5}{c}{Rozmiar warstwy ukrytej} \\\\
 &   32 & 64 & 128 & 256 & 512 & \\\\
\\hline
    """

    end = """
\\hline
\\end{tabular}
\\label{tab:atnn_hidden_sizes}
\\end{table}
    """

    results = """
\\multirow{1}{*}{\\makecell{electricity}} & """ + d["elec"][1] + """ & """ + d["elec"][2] + """ & """ + d["elec"][3] + """ & """ + d["elec"][4] + """ & """ + d["elec"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & """ + d["weather_norm"][1] + """ & """ + d["weather_norm"][2] + """ & """ + d["weather_norm"][3] + """ & """ + d["weather_norm"][4] + """ & """ + d["weather_norm"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & """ + d["mnist_abrupt_atnn_like"][1] + """ & """ + d["mnist_abrupt_atnn_like"][2] + """ & """ + d["mnist_abrupt_atnn_like"][3] + """ & """ + d["mnist_abrupt_atnn_like"][4] + """ & """ + d["mnist_abrupt_atnn_like"][5] + """ \\\\

\\multirow{1}{*}{\\makecell{fashion\\textsubscript{abr}}} & """ + d["fashion_abr"][1] + """ & """ + d["fashion_abr"][2] + """ & """ + d["fashion_abr"][3] + """ & """ + d["fashion_abr"][4] + """ & """ + d["fashion_abr"][5] + """ \\\\
    
    """

    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    vals = {}
    printables = {}
    for dataset in df["Dataset"].unique():
        print("dataset: ", dataset)

        keys_to_sizes = {
            1: "_32",
            2: "_64",
            3: "128",
            4: "256",
            5: "512"
        }

        vals[dataset] = {}
        vals[dataset]["_32"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls32_lambda5000_gamma1.0")]
        vals[dataset]["_64"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls64_lambda5000_gamma1.0")]
        vals[dataset]["128"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls128_lambda5000_gamma1.0")]
        vals[dataset]["256"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma1.0")]
        vals[dataset]["512"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls512_lambda5000_gamma1.0")]

        vals[dataset]["_32"] = float(vals[dataset]["_32"]["Accuracy"].values[0])
        vals[dataset]["_64"] = float(vals[dataset]["_64"]["Accuracy"].values[0])
        vals[dataset]["128"] = float(vals[dataset]["128"]["Accuracy"].values[0])
        vals[dataset]["256"] = float(vals[dataset]["256"]["Accuracy"].values[0])
        vals[dataset]["512"] = float(vals[dataset]["512"]["Accuracy"].values[0])

        printables[dataset] = {}
        printables[dataset][1] = "{:.2f}".format(vals[dataset]["_32"]).replace(".", ",")
        printables[dataset][2] = "{:.2f}".format(vals[dataset]["_64"]).replace(".", ",")
        printables[dataset][3] = "{:.2f}".format(vals[dataset]["128"]).replace(".", ",")
        printables[dataset][4] = "{:.2f}".format(vals[dataset]["256"]).replace(".", ",")
        printables[dataset][5] = "{:.2f}".format(vals[dataset]["512"]).replace(".", ",")


        for key in printables[dataset].keys():
            if vals[dataset][keys_to_sizes[key]] == max(vals[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

        results[dataset] = printables[dataset]

    return results


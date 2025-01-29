import numpy as np
import pandas as pd


def atnn_lr_accuracy_comparison_table(df: pd.DataFrame):
    d = __printable_accuracies(df)

    caption = """
\\begin{table}[H]
\\centering
\\caption{Dokładność ze względu na szybkość uczenia $\\eta$. Podczas eksperymentów szerokość warstwy ukrytej ustawiono na 256 neuronów, a parametr regularyzacji $\\lambda=5000$.}
\\begin{tabular}{lcccccc}
\\hline
\\multicolumn{1}{c}{\\textbf{Zbiór danych}} & \multicolumn{6}{c}{\\textbf{Dokładność}} \\\\ 
 &  \\multicolumn{6}{c}{Początkowa szybkość uczenia $\\eta$} \\\\
 &   $1\\cdot10^{-1}$ & $1\\cdot10^{-2}$ & $2\\cdot10^{-2}$ & $5\\cdot10^{-2}$ & $1\\cdot10^{-3}$ & $1\\cdot10^{-4}$ \\\\
\\hline
    """

    end = """
\\hline
\\end{tabular}
\\label{tab:atnn_learing_rates}
\\end{table}
    """

    results = """
\\multirow{1}{*}{\\makecell{electricity}} & """ + d["elec"][1] + """ & """ + d["elec"][2] + """ & """ + d["elec"][3] + """ & """ + d["elec"][4] + """ & """ + d["elec"][5] + """& """ + d["elec"][6] + """ \\\\

\\multirow{1}{*}{\\makecell{weather}} & """ + d["weather_norm"][1] + """ & """ + d["weather_norm"][2] + """ & """ + d["weather_norm"][3] + """ & """ + d["weather_norm"][4] + """ & """ + d["weather_norm"][5] + """& """ + d["weather_norm"][6] + """ \\\\

\\multirow{1}{*}{\\makecell{MNIST\\textsubscript{abr}}} & """ + d["mnist_abrupt_atnn_like"][1] + """ & """ + d["mnist_abrupt_atnn_like"][2] + """ & """ + d["mnist_abrupt_atnn_like"][3] + """ & """ + d["mnist_abrupt_atnn_like"][4] + """ & """ + d["mnist_abrupt_atnn_like"][5] + """& """ + d["mnist_abrupt_atnn_like"][6] + """ \\\\

\\multirow{1}{*}{\\makecell{fashion\\textsubscript{abr}}} & """ + d["fashion_abr"][1] + """ & """ + d["fashion_abr"][2] + """ & """ + d["fashion_abr"][3] + """ & """ + d["fashion_abr"][4] + """ & """ + d["fashion_abr"][5] + """& """ + d["fashion_abr"][6] + """ \\\\
    
    """

    text = caption + results + end
    return text


def __printable_accuracies(df: pd.DataFrame) -> dict:
    results = {}
    vals = {}
    printables = {}
    for dataset in df["Dataset"].unique():
        print("dataset: ", dataset)

        keys_to_lr = {
            1: "1E-1",
            2: "5E-2",
            3: "2E-2",
            4: "1E-2",
            5: "1E-3",
            6: "1E-4",
        }

        vals[dataset] = {}
        vals[dataset]["1E-1"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.1_hls256_lambda5000_gamma1.0")]
        vals[dataset]["5E-2"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.05_hls256_lambda5000_gamma1.0")]
        vals[dataset]["2E-2"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma1.0")]
        vals[dataset]["1E-2"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.01_hls256_lambda5000_gamma1.0")]
        vals[dataset]["1E-3"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.001_hls256_lambda5000_gamma1.0")]
        vals[dataset]["1E-4"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr1.0E-4_hls256_lambda5000_gamma1.0")]

        vals[dataset]["1E-1"] = float(vals[dataset]["1E-1"]["Accuracy"].values[0])
        vals[dataset]["5E-2"] = float(vals[dataset]["5E-2"]["Accuracy"].values[0])
        vals[dataset]["2E-2"] = float(vals[dataset]["2E-2"]["Accuracy"].values[0])
        vals[dataset]["1E-2"] = float(vals[dataset]["1E-2"]["Accuracy"].values[0])
        vals[dataset]["1E-3"] = float(vals[dataset]["1E-3"]["Accuracy"].values[0])
        vals[dataset]["1E-4"] = float(vals[dataset]["1E-4"]["Accuracy"].values[0])

        printables[dataset] = {}
        printables[dataset][1] = "{:.2f}".format(vals[dataset]["1E-1"]).replace(".", ",")
        printables[dataset][2] = "{:.2f}".format(vals[dataset]["5E-2"]).replace(".", ",")
        printables[dataset][3] = "{:.2f}".format(vals[dataset]["2E-2"]).replace(".", ",")
        printables[dataset][4] = "{:.2f}".format(vals[dataset]["1E-2"]).replace(".", ",")
        printables[dataset][5] = "{:.2f}".format(vals[dataset]["1E-3"]).replace(".", ",")
        printables[dataset][6] = "{:.2f}".format(vals[dataset]["1E-4"]).replace(".", ",")


        for key in printables[dataset].keys():
            if vals[dataset][keys_to_lr[key]] == max(vals[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

        results[dataset] = printables[dataset]

    return results


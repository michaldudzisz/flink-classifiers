import numpy as np
import pandas as pd


def atnn_lambda_accuracy_comparison_table(df: pd.DataFrame):
    d = __printable_accuracies(df)

    caption = """
\\begin{table}[H]
\\centering
\\caption{Dokładność ze względu na parametr regularyzacji $\\lambda$. Podczas eksperymentów szybkość uczenia ustawiono na wartość $\\eta=0,02$, a szerokość warstwy ukrytej ustawiono na 256 neuronów.}
\\begin{tabular}{lccccc}
\\hline
\\multicolumn{1}{c}{\\textbf{Zbiór danych}} & \multicolumn{5}{c}{\\textbf{Dokładność}} \\\\ 
 &  $\\lambda=0$ & $\\lambda=50$ & $\\lambda=500$ & $\\lambda=5000$ & $\\lambda=50000$  \\\\
\\hline
    """

    end = """
\\hline
\\end{tabular}
\\label{tab:atnn_lambdas}
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

        keys_to_lambdas = {
            1: "____0",
            2: "___50",
            3: "__500",
            4: "_5000",
            5: "50000"
        }

        vals[dataset] = {}
        vals[dataset]["____0"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda0_gamma1.0")]
        vals[dataset]["___50"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda50_gamma1.0")]
        vals[dataset]["__500"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda500_gamma1.0")]
        vals[dataset]["_5000"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda5000_gamma1.0")]
        vals[dataset]["50000"] = df[(df["Dataset"] == dataset) & (df["Method"] == "atnn") & (df["Params"] == "lr0.02_hls256_lambda50000_gamma1.0")]

        vals[dataset]["____0"] = float(vals[dataset]["____0"]["Accuracy"].values[0])
        vals[dataset]["___50"] = float(vals[dataset]["___50"]["Accuracy"].values[0])
        vals[dataset]["__500"] = float(vals[dataset]["__500"]["Accuracy"].values[0])
        vals[dataset]["_5000"] = float(vals[dataset]["_5000"]["Accuracy"].values[0])
        vals[dataset]["50000"] = float(vals[dataset]["50000"]["Accuracy"].values[0])

        printables[dataset] = {}
        printables[dataset][1] = "{:.2f}".format(vals[dataset]["____0"]).replace(".", ",")
        printables[dataset][2] = "{:.2f}".format(vals[dataset]["___50"]).replace(".", ",")
        printables[dataset][3] = "{:.2f}".format(vals[dataset]["__500"]).replace(".", ",")
        printables[dataset][4] = "{:.2f}".format(vals[dataset]["_5000"]).replace(".", ",")
        printables[dataset][5] = "{:.2f}".format(vals[dataset]["50000"]).replace(".", ",")


        for key in printables[dataset].keys():
            if vals[dataset][keys_to_lambdas[key]] == max(vals[dataset].values()):
                printables[dataset][key] = f"\\textbf{{{printables[dataset][key]}}}"

        results[dataset] = printables[dataset]

    return results


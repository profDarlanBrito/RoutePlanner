import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_csv_data_build(csv_path, basename):

    # Carregar o arquivo CSV
    df = pd.read_csv(csv_path)

    # Filtragem dos dados conforme fornecido
    own_west = df[(df["ply"] == "West_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_west["method"] = "Own"

    own_chinese = df[(df["ply"] == "Chinese_Old_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_chinese["method"] = "Own"

    own_old = df[(df["ply"] == "Old_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_old["method"] = "Own"

    tmax = own_west["distance"].mean() + own_chinese["distance"].mean() + own_old["distance"].mean()
    # tmax = int(basename.split("_")[-1])

    own_west["tmax"] = tmax
    own_chinese["tmax"] = tmax
    own_old["tmax"] = tmax

    return own_west, own_chinese, own_old


def plot_images_build(west_data, chinese_data, old_data):
    # Configuração do estilo do Seaborn
    sns.set_theme(style="whitegrid")

    # Calcular a média de rmse por tmax para cada conjunto de dados
    west_mean = west_data.groupby("tmax", as_index=False)["rmse"].mean()
    chinese_mean = chinese_data.groupby("tmax", as_index=False)["rmse"].mean()
    old_mean = old_data.groupby("tmax", as_index=False)["rmse"].mean()

    # Criar a figura
    plt.figure(figsize=(7, 14 / 3))

    # Adicionar cada conjunto de dados como uma linha
    sns.lineplot(data=chinese_mean, x="tmax", y="rmse", label=r"$O_1$", linewidth=2, marker="s")
    sns.lineplot(data=west_mean, x="tmax", y="rmse", label=r"$O_2$", linewidth=2, marker="o")
    sns.lineplot(data=old_mean, x="tmax", y="rmse", label=r"$O_3$", linewidth=2, marker="d")

    # Configuração dos rótulos e título
    plt.xlabel("$T_{max}$", fontsize=14)
    plt.ylabel("Mean Reconstruction Error ($Er$, meters)", fontsize=14)
    # plt.title("Comparação da Média de RMSE entre Cenários de Construção", fontsize=16)

    # Legenda com suporte a LaTeX
    plt.legend(title="Objects", fontsize=12)

    # Melhorar layout
    plt.tight_layout()

    # Exibir o gráfico
    # plt.show()

    # Salvar figura
    plt.savefig(f"expr_images/rmse_x_tmax.png", dpi=600)


csvs = {
    # "CA_4000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/1/metrics.csv",
    # "CA_6000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/dalek/build/1/metrics.csv",
    "CA_5000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/dalek/build/0/metrics.csv",
    "CA_5000_Tmax_900": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/0/metrics.csv",
    "CA_5000_Tmax_800": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/2/metrics.csv",
    # "CA_5000_Tmax_850": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/4/metrics.csv",
}

scene_data = [get_csv_data_build(csv_path, basename) for basename, csv_path in csvs.items()]


own_west = []
own_chinese = []
own_old = []
for own_west_t, own_chinese_t, own_old_t in scene_data:
    own_west.append(own_west_t)
    own_chinese.append(own_chinese_t)
    own_old.append(own_old_t)


west_data = pd.concat(own_west)
chinese_data = pd.concat(own_chinese)
old_data = pd.concat(own_old)

print(west_data, chinese_data, old_data)

plot_images_build(west_data, chinese_data, old_data)

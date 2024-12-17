import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_csv_data_build(csv_path):

    # Carregar o arquivo CSV
    df = pd.read_csv(csv_path)

    # Filtragem dos dados conforme fornecido
    own_scene = df[~df["reconstruction_path"].str.contains("_exp")]
    own_scene["method"] = "Own"

    roberts_scene = df[df["reconstruction_path"].str.contains("op_exp")]
    roberts_scene["method"] = "Roberts"

    random_scene = df[df["reconstruction_path"].str.contains("random_exp")]
    random_scene["method"] = "Random"

    spiral_scene = df[df["reconstruction_path"].str.contains("spiral_exp")]
    spiral_scene["method"] = "Spiral"

    own_west = df[(df["ply"] == "West_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_west["method"] = "Own"

    own_chinese = df[(df["ply"] == "Chinese_Old_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_chinese["method"] = "Own"

    own_old = df[(df["ply"] == "Old_Build.ply") & (~df["reconstruction_path"].str.contains("_exp"))]
    own_old["method"] = "Own"

    roberts_west = df[(df["ply"] == "West_Build.ply") & (df["reconstruction_path"].str.contains("op_exp"))]
    roberts_west["method"] = "Roberts"

    roberts_chinese = df[(df["ply"] == "Chinese_Old_Build.ply") & (df["reconstruction_path"].str.contains("op_exp"))]
    roberts_chinese["method"] = "Roberts"

    roberts_old = df[(df["ply"] == "Old_Build.ply") & (df["reconstruction_path"].str.contains("op_exp"))]
    roberts_old["method"] = "Roberts"

    random_west = df[(df["ply"] == "West_Build.ply") & (df["reconstruction_path"].str.contains("random_exp"))]
    random_west["method"] = "Random"

    random_chinese = df[(df["ply"] == "Chinese_Old_Build.ply") & (df["reconstruction_path"].str.contains("random_exp"))]
    random_chinese["method"] = "Random"

    random_old = df[(df["ply"] == "Old_Build.ply") & (df["reconstruction_path"].str.contains("random_exp"))]
    random_old["method"] = "Random"

    spiral_west = df[(df["ply"] == "West_Build.ply") & (df["reconstruction_path"].str.contains("spiral_exp"))]
    spiral_west["method"] = "Spiral"

    spiral_chinese = df[(df["ply"] == "Chinese_Old_Build.ply") & (df["reconstruction_path"].str.contains("spiral_exp"))]
    spiral_chinese["method"] = "Spiral"

    spiral_old = df[(df["ply"] == "Old_Build.ply") & (df["reconstruction_path"].str.contains("spiral_exp"))]
    spiral_old["method"] = "Spiral"

    # Concatenando todos os dados
    old_data = pd.concat([own_old, roberts_old, spiral_old])
    west_data = pd.concat([own_west, roberts_west, spiral_west])
    chinese_data = pd.concat([own_chinese, roberts_chinese, spiral_chinese])
    scene_data = pd.concat([own_scene, roberts_scene, spiral_scene])

    # print(spiral_old['rmse'].mean())
    # print(spiral_west['rmse'].mean())
    # print(spiral_chinese['rmse'].mean())

    return scene_data, old_data, west_data, chinese_data


def plot_images_build(basename, scene_data, old_data, west_data, chinese_data):

    for data, obj_name in zip([old_data, west_data, chinese_data, scene_data], ["old", "west", "chinese", "scene"]):
        # Plotando o boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, x="method", y="rmse", palette="Set2")

        # Personalizando o gráfico
        plt.xlabel("Method", fontsize=12)
        plt.ylabel("Reconstruction Error ($Er$, meteres)", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Exibindo o gráfico
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"expr_images/{basename}_{obj_name}.png", dpi=600)


csvs = {
    "CA_5000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/dalek/build/0/metrics.csv",
    "CA_6000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/dalek/build/1/metrics.csv",
    "CA_5000_Tmax_900": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/0/metrics.csv",
    "CA_5000_Tmax_800": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/2/metrics.csv",
    "CA_4000_Tmax_1000": "/mnt/e/Arquivos/wsl_arquivos/experiments_artigo/wall-e/build/1/metrics.csv",
}

for basename, csv_path in csvs.items():
    scene_data, old_data, west_data, chinese_data = get_csv_data_build(csv_path)
    plot_images_build(basename, scene_data, old_data, west_data, chinese_data)

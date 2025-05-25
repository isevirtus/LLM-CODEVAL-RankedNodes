import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# =========================
# Carrega o CSV com os dados
# =========================
df = pd.read_csv("resultado_comparacao_brier.csv")

# Converte as colunas de string para lista numérica
df["modelo_probs"] = df["modelo_probs"].apply(ast.literal_eval)
df["agena_probs"] = df["agena_probs"].apply(ast.literal_eval)

# =========================
# Gráfico 1 – Brier Score por cenário
# =========================
plt.figure(figsize=(14, 5))
plt.plot(df["id"], df["brier_score"], marker='o', linestyle='-', color='navy', label='Brier Score')
plt.axhline(y=0.001, color='red', linestyle='--', label='Reference: 0.001')
plt.title("Brier Score Error per Scenario – Our Model vs AgenaRisk")
plt.xlabel("Scenario ID")
plt.ylabel("Brier Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_brier_score_linha.png")
plt.show()

# =========================
# Gráfico 2 – Comparação de distribuições (TNormal)
# =========================
cenarios_para_plotar = [1, 15, 28, 48]  # Edite essa lista com os IDs que quiser visualizar

estados = ['VL', 'L', 'M', 'H', 'VH']
x = np.arange(len(estados))

for cid in cenarios_para_plotar:
    linha = df[df["id"] == cid].iloc[0]
    modelo_probs = linha["modelo_probs"]
    agena_probs = linha["agena_probs"]

    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, modelo_probs, width=0.4, label="Our model", color='navy')
    plt.bar(x + 0.2, agena_probs, width=0.4, label="AgenaRisk", color='#D3D3D3')

    plt.xticks(x, estados)
    plt.ylim(0, 1)
    plt.title(f"Scenario {cid} – Probability Distributions")
    plt.xlabel("States")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"distribuicao_cenario_{cid}.png")
    plt.show()

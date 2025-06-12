import numpy as np
import pandas as pd
from datetime import date, timedelta
from pulp import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simulando dados
dados = {
    "Sala": ["01", "02", "03", "01", "02", "03"],
    "Dia": ["Segunda", "Segunda", "Segunda", "Terça", "Terça", "Terça"],
    "Tempo_médio": [50, 45, 60, 48, 60, 40],
    "Tempo_total": [600] * 6
}
df = pd.DataFrame(dados)

# Lista para armazenar os resultados
resultados = []

# Loop de otimização para cada linha (sala + dia)
for i, row in df.iterrows():
    sala = row["Sala"]
    dia = row["Dia"]
    d = row["Tempo_médio"]
    T = row["Tempo_total"]

    # Criar o modelo
    prob = LpProblem(f"Otimiza_{sala}_{dia}", LpMinimize)

    # Variável de decisão: número de atendimentos
    x = LpVariable(f"Atendimentos_{sala}_{dia}", lowBound=8, cat=LpInteger)

    # Função objetivo: minimizar tempo ocioso
    prob += T - (d * x)

    # Restrição: tempo ocupado não pode passar do total
    prob += d * x <= T

    # Resolver
    prob.solve()

    # Guardar resultados
    resultados.append({
        "Sala": sala,
        "Dia": dia,
        "Tempo_médio": d,
        "Atendimentos_ótimos": int(x.varValue),
        "Tempo_ocupado": d * int(x.varValue),
        "Tempo_ocioso": T - (d * int(x.varValue)),
        "Status": LpStatus[prob.status]
    })

# Criar DataFrame de resultado
df_resultado = pd.DataFrame(resultados)

# Visualizar resultado
print(df_resultado)

# --- VISUALIZAÇÃO COM PLOTLY ---
# Agrupar por dia para gráfico
dias = df_resultado["Dia"].unique()
fig = make_subplots(rows=1, cols=1, subplot_titles=["Tempo Ocioso por Sala"])

for dia in dias:
    df_dia = df_resultado[df_resultado["Dia"] == dia]
    fig.add_trace(go.Bar(
        x=df_dia["Sala"],
        y=df_dia["Tempo_ocioso"],
        name=f"{dia}"
    ))

fig.update_layout(
    title="Comparativo de Tempo Ocioso por Sala e Dia",
    xaxis_title="Sala",
    yaxis_title="Tempo Ocioso (min)",
    barmode='group'
)
fig.show()

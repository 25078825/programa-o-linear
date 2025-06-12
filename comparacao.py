!pip install 
-q pulp
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pulp import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parâmetros de Configuração e Geração de Dados (Baseado no seu pip install.docx) ---
# Define a semente para reprodutibilidade dos resultados
np.random.seed(42)

num_rooms = 3 # Número de salas para o exemplo
num_professionals = 5 # Número de profissionais para o exemplo
start_date = date(2024, 6, 10)
end_date = date(2024, 6, 12) # Três dias para o exemplo

min_room_availability = 500 # minutos
max_room_availability = 600 # minutos
min_session_duration_original = 40 # minutos (duração mínima de sessão para dados originais)
max_session_duration_original = 60 # minutos (duração máxima de sessão para dados originais)
min_appointments_per_room_day = 8 # mínimo de atendimentos por sala/dia (restrição)

# Parâmetros para a variável de decisão 'd_med' (duração média otimizada)
min_session_duration_optimized = 30 # Limite inferior para d_med na otimização
max_session_duration_optimized = 70 # Limite superior para d_med na otimização

# Gera IDs para salas e profissionais
room_ids = [f'Sala_{i:02d}' for i in range(1, num_rooms + 1)]
professional_ids = [f'Profissional_{i:02d}' for i in range(1, num_professionals + 1)]

# Gera todas as datas no intervalo
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += timedelta(days=1)

# Geração de Dados para Salas (rooms dictionary) - representa o cenário "original" antes da otimização da duração
# 'times_used' aqui será o número de atendimentos que já estavam previstos ou observados
rooms_original_data = {}
for room_id in room_ids:
    rooms_original_data[room_id] = {}
    for dt in dates:
        available_time = np.random.randint(min_room_availability, max_room_availability + 1)
        
        # Simula uma duração média "original" para esta sala/dia
        original_avg_session_duration = np.random.randint(min_session_duration_original, max_session_duration_original + 1)
        
        # Calcula um número de atendimentos "original" que se encaixa na duração média original
        # Garante pelo menos min_appointments_per_room_day
        times_used_original = max(min_appointments_per_room_day, int(available_time / original_avg_session_duration * np.random.uniform(0.7, 0.9)))
        
        # Ajusta para não exceder o tempo disponível com a duração original
        if (times_used_original * original_avg_session_duration) > available_time:
             times_used_original = int(available_time / original_avg_session_duration)

        rooms_original_data[room_id][dt] = {
            'available_time': available_time,
            'times_used': times_used_original,
            'original_avg_session_duration': original_avg_session_duration # Duração média original para comparação
        }

# Geração de Dados para Profissionais (simplificado, não usado na otimização de tempo ocioso de sala aqui)
professionals_data = {}
for prof_id in professional_ids:
    professionals_data[prof_id] = {}
    for dt in dates:
        professionals_data[prof_id][dt] = {
            'available_time': np.random.randint(300, 480),
            'sessions_assigned': np.random.randint(5, 10)
        }

print("--- Dados Simulados Originais Gerados (rooms_original_data) ---")
for room_id in room_ids:
    for dt in dates:
        data = rooms_original_data[room_id][dt]
        print(f"Sala: {room_id}, Data: {dt}, Disponível: {data['available_time']} min, "
              f"Atendimentos Originais: {data['times_used']}, "
              f"Duração Média Original: {data['original_avg_session_duration']} min")

# --- 2. Otimização com PuLP e Comparação ---

# Lista para armazenar os resultados da otimização
results_optimization = []

# Dicionário para armazenar o tempo ocioso de cada sala/data antes da otimização
original_idle_times = {}

for room in room_ids:
    original_idle_times[room] = {}
    for date in dates:
        # Calcular tempo ocioso original
        available_time = rooms_original_data[room][date]['available_time']
        times_used = rooms_original_data[room][date]['times_used']
        original_avg_session = rooms_original_data[room][date]['original_avg_session_duration']
        
        # Tempo ocioso antes da otimização (com a duração média original)
        original_idle_time = available_time - (original_avg_session * times_used)
        original_idle_times[room][date] = original_idle_time

        # --- Modelo de Otimização PuLP ---
        prob = LpProblem(f"Otimiza_Sala_{room}_Data_{date}", LpMinimize)

        # Variável de decisão: d_med (Duração Média Otimizada por Atendimento)
        # LowBound/UpBound são os limites da duração média de sessão que o otimizador pode escolher
        d_med = LpVariable("DuracaoMediaOtimizada", lowBound=min_session_duration_optimized, 
                           upBound=max_session_duration_optimized, cat='Integer')

        # Função objetivo: Minimizar o tempo ocioso
        # rooms_original_data[room][date]['times_used'] é o número de atendimentos que precisa ser acomodado
        prob += available_time - (d_med * times_used), "Tempo Ocioso Total"

        # Restrição: o tempo ocupado não pode exceder o tempo disponível da sala
        prob += d_med * times_used <= available_time, "Limite Tempo Disponível"
        
        # Outras restrições que podem ser adicionadas conforme o docx (ex: max_possible_sessions_per_room_day)
        # Neste modelo específico, 'times_used' é um dado de entrada, não uma variável de decisão.
        # Se 'times_used' fosse uma variável, teríamos mais restrições.
        # A restrição min_appointments_per_room_day é garantida na geração dos dados originais.

        # Resolver o problema
        status = prob.solve()

        # Coletar resultados
        optimized_d_med = value(d_med) if d_med.varValue is not None else original_avg_session # Caso não encontre ótimo, usa o original
        optimized_idle_time = value(prob.objective) if prob.objective is not None else original_idle_time
        
        results_optimization.append({
            "Sala": room,
            "Data": date,
            "Tempo_Disponivel_min": available_time,
            "Total_Atendimentos": times_used, # Este é o número de atendimentos a ser acomodado
            "Duracao_Media_Original_min": original_avg_session,
            "Tempo_Ocioso_Original_min": original_idle_time,
            "Duracao_Media_Otimizada_min": optimized_d_med,
            "Tempo_Ocioso_Otimizado_min": optimized_idle_time,
            "Status_Otimizacao": LpStatus[status]
        })

# Criar DataFrame com os resultados
df_final_comparison = pd.DataFrame(results_optimization)

print("\n--- Resultados Detalhados da Otimização por Sala e Data ---")
print(df_final_comparison)

# --- Análise e Comprovação da Otimização ---
total_idle_original = df_final_comparison["Tempo_Ocioso_Original_min"].sum()
total_idle_optimized = df_final_comparison["Tempo_Ocioso_Otimizado_min"].sum()

print("\n--- Sumário de Comprovação da Otimização ---")
print(f"Tempo Ocioso Total (Cenário Original): {total_idle_original:.2f} minutos")
print(f"Tempo Ocioso Total (Cenário Otimizado pelo seu Método): {total_idle_optimized:.2f} minutos")

if total_idle_optimized < total_idle_original:
    reduction_percentage = ((total_idle_original - total_idle_optimized) / total_idle_original) * 100
    print(f"**COMPROVADO: Sua ideia faz sentido e é otimizada!**")
    print(f"Houve uma **redução de {reduction_percentage:.2f}%** no tempo ocioso total.")
    print(f"Isso significa que o ajuste na duração média dos atendimentos (`Duracao_Media_Otimizada_min`)")
    print(f"permite um uso mais eficiente das salas, minimizando o tempo sem uso.")
elif total_idle_optimized == total_idle_original:
    print("Para este conjunto de dados, o método de otimização manteve o tempo ocioso igual ao original.")
    print("Isso pode acontecer se o cenário original já for ótimo dentro das restrições.")
else:
    increase_percentage = ((total_idle_optimized - total_idle_original) / total_idle_original) * 100
    print(f"ATENÇÃO: O tempo ocioso aumentou em {increase_percentage:.2f}%. Isso indica que as restrições ou o modelo precisam ser revisados.")

# --- Opcional: Salvar os resultados em um CSV ---
df_final_comparison.to_csv("comprovacao_otimizacao_completa.csv", index=False)
print("\nResultados detalhados da comprovação salvos em 'comprovacao_otimizacao_completa.csv'")

# --- Visualização do Tempo Ocioso (opcional, para insights) ---
fig = make_subplots(rows=num_rooms, cols=1, 
                    subplot_titles=[f'Tempo Ocioso Sala {room.split("_")[1]}' for room in room_ids])

for i, room in enumerate(room_ids):
    df_room = df_final_comparison[df_final_comparison["Sala"] == room]
    fig.add_trace(go.Bar(
        x=df_room["Data"],
        y=df_room["Tempo_Ocioso_Original_min"],
        name=f"Original - {room}",
        marker_color='skyblue'
    ), row=i+1, col=1)
    fig.add_trace(go.Bar(
        x=df_room["Data"],
        y=df_room["Tempo_Ocioso_Otimizado_min"],
        name=f"Otimizado - {room}",
        marker_color='lightcoral'
    ), row=i+1, col=1)

fig.update_layout(title_text="Comparativo de Tempo Ocioso: Original vs. Otimizado por Sala e Data", height=400 * num_rooms)
fig.show()
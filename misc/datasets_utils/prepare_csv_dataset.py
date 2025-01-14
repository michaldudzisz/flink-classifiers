import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Wczytanie pliku CSV
input_file = 'datasets/weather.csv'  # Zmień na ścieżkę do swojego pliku
output_file = 'datasets/weather_norm.csv'  # Zmień na ścieżkę do zapisu wynikowego pliku

df = pd.read_csv(input_file).head(100000)

# Normalizacja kolumn do zakresu (0, 1) z wykluczeniem ostatniej kolumny
scaler = MinMaxScaler()
df_to_normalize = df.iloc[:, :-1]
df_normalized = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=df_to_normalize.columns)

# Dodanie ostatniej, nienormalizowanej kolumny jako int
df_normalized[df.columns[-1]] = df[df.columns[-1]].astype(int).values

# Przygotowanie danych do zapisu
columns = df.columns
rows = []
header = "feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,feat_7,feat_8,target"
rows.append(header)
for index, row in df_normalized.iterrows():
    row_values = row.tolist()
    formatted_row = "#".join(map(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), row_values[:-1])) + "," + str(int(row_values[-1]))
    rows.append(formatted_row)

# Zapis do pliku
with open(output_file, 'w') as f:
    for line in rows:
        f.write(line + '\n')

print(f'Plik wynikowy zapisany jako: {output_file}')

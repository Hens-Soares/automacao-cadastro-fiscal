"""
Automação de Processamento Fiscal e Gerencial (Polars + Google Drive API)
-------------------------------------------------------------------------
Descrição: Leitura, filtragem e transformação de grandes volumes de dados 
           fiscais utilizando Polars (Rust engine) com integração ao Google Drive 
           e Google Sheets.
"""

import os
import shutil
import pandas as pd
import polars as pl
import gspread
from google.colab import drive, auth
from google.auth import default

# ==============================================================================
# CONFIGURAÇÕES (Substitua pelos caminhos/IDs do seu ambiente)
# ==============================================================================
# ID da planilha Google Sheets de apoio (obtido na URL da planilha)
SHEETS_CONF_ID = os.getenv("SHEETS_CONF_ID", "SEU_SPREADSHEET_ID_AQUI")
NOME_ABA_CONF = "FL_UF"

# Diretórios do Google Drive e Temp SSD
PASTA_ORIGEM = "/content/drive/MyDrive/SEU_DIRETORIO/ARQUIVOS"
PASTA_DESTINO = "/content/drive/MyDrive/SEU_DIRETORIO/RESULTADO"
PASTA_TEMP = "/content/temp_processing"
NOME_ARQUIVO_SAIDA = "LAYOUT_CADASTRO.csv"

# ==============================================================================
# INICIALIZAÇÃO E AUTENTICAÇÃO
# ==============================================================================
print("🔑 Autenticando no Google Cloud / Drive...")
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

drive.mount('/content/drive')

os.makedirs(PASTA_TEMP, exist_ok=True)
os.makedirs(PASTA_DESTINO, exist_ok=True)

# ==============================================================================
# 1. CARREGAR TABELA DE APOIO (FL_UF) VIA GOOGLE SHEETS API
# ==============================================================================
print("📥 Carregando tabela de mapeamento FL_UF...")
de_para_uf = {}

try:
    sh = gc.open_by_key(SHEETS_CONF_ID)
    worksheet = sh.worksheet(NOME_ABA_CONF)
    dados_uf = worksheet.get_all_values()
    
    df_fl_uf = pd.DataFrame(dados_uf[1:], columns=dados_uf[0])
    col_chave_uf = df_fl_uf.columns[0]
    col_valor_uf = df_fl_uf.columns[1]

    chave_clean = df_fl_uf[col_chave_uf].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    valor_clean = df_fl_uf[col_valor_uf].astype(str).str.strip()
    
    de_para_uf = dict(zip(chave_clean, valor_clean))
    print(f"   ✓ Mapeamento FL_UF carregado com sucesso! ({len(de_para_uf)} registros)")
except Exception as e:
    print(f"   ❌ Erro ao acessar a planilha de apoio: {e}")

# ==============================================================================
# 2. PROCESSAMENTO DE ARQUIVOS
# ==============================================================================
arquivos_processados = 0

for nome_arquivo in os.listdir(PASTA_ORIGEM):
    if nome_arquivo.endswith(('.xlsx', '.xls', '.csv')):
        print(f"\n⚡ Processando arquivo com Polars (Rust): {nome_arquivo}")
        
        caminho_origem = os.path.join(PASTA_ORIGEM, nome_arquivo)
        caminho_temp_origem = os.path.join(PASTA_TEMP, nome_arquivo)
        
        # Copia para SSD local do Colab para máxima velocidade
        shutil.copy(caminho_origem, caminho_temp_origem)
        
        # --- LEITURA RÁPIDA ---
        if nome_arquivo.endswith('.csv'):
            df = pl.read_csv(
                caminho_temp_origem, 
                infer_schema_length=0,
                truncate_ragged_lines=True,
                ignore_errors=True
            )
        else:
            df = pl.read_excel(caminho_temp_origem, engine="calamine")
            df = df.select([pl.col(col).cast(pl.Utf8) for col in df.columns])

        colunas = df.columns
        
        # Mapeamento de colunas dinâmicas por nome/índice
        col_Y_filtro = "TRIB" if "TRIB" in colunas else colunas[24]
        col_AL_filtro = colunas[37]

        # Tratamento e Filtragem Inicial (Trib 702/703 e AL = 0)
        df_limpo = df.with_columns([
            pl.col(col_Y_filtro).str.strip_chars().str.replace(r"\.0$", "").alias("temp_Y"),
            pl.col(col_AL_filtro).str.strip_chars().str.replace(r"\.0$", "").alias("temp_AL")
        ])

        valores_Y_validos = ["702", "703", "0702", "0703"]
        valores_AL_validos = ["0", "00", "0.0"]

        df_filtrado = df_limpo.filter(
            pl.col("temp_Y").is_in(valores_Y_validos) & 
            pl.col("temp_AL").is_in(valores_AL_validos)
        ).drop(["temp_Y", "temp_AL"])

        print(f"   📊 Registros filtrados: {df_filtrado.height} linhas")

        # --- SELEÇÃO E MAPEAMENTO DE POSIÇÕES ---
        col_A  = colunas[0]   # 1ª coluna (MERCADORIA)
        col_K  = colunas[10]  # 11ª coluna (K - ICM.SAI.BS)
        col_L  = colunas[11]  # 12ª coluna (L - ICM.SAI.EF)
        col_P  = colunas[15]  # 16ª coluna (P - ICM.POB.CON)
        col_Y  = colunas[24]  # 25ª coluna (TRIB)
        col_AI = colunas[34]  # 35ª coluna (AI - PRC.PAUTA AT)
        col_AT = colunas[45]  # 46ª coluna (COMPRADOR)
        col_AU = colunas[46]  # 47ª coluna (NOME COMPRADOR)
        col_AV = colunas[47]  # 48ª coluna (COMPR.MATRIZ)
        col_AW = colunas[48]  # 49ª coluna (NOME COMPRADOR)
        col_AX = colunas[49]  # 50ª coluna (AX - Chave da UF)

        # 1. Procv da UF
        chaves_ax = df_filtrado[col_AX].str.strip_chars().str.replace(r"\.0$", "").to_list()
        uf_mapeada = [de_para_uf.get(k, "") for k in chaves_ax]

        # 2. Conversão e Cálculo das Colunas C e D
        df_transformado = df_filtrado.with_columns([
            pl.Series("UF", uf_mapeada),
            pl.col(col_K).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).alias("_k_num"),
            pl.col(col_L).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).alias("_l_num"),
            pl.col(col_P).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).alias("_p_num"),
            pl.col(col_AI).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).alias("_pauta_num")
        ]).with_columns([
            (pl.col("_k_num") + pl.col("_p_num")).round(2).alias("_col_C_num"),
            (pl.col("_l_num") + pl.col("_p_num")).round(2).alias("_col_D_num")
        ])

        # 3. Lógica para Coluna G (Cesta ?) baseada nas novas Colunas C e D
        cesta_condicao = (
            pl.col("_col_D_num").is_in([2.0, 4.0, 14.71]) | 
            ((pl.col("_col_C_num") != 7.0) & (pl.col("_col_D_num") == 7.0))
        )

        # 4. Lógica para Coluna H (IND. Reducao) baseada nas novas Colunas C e D
        reducao_condicao = (pl.col("_col_C_num") != pl.col("_col_D_num"))

        # 5. Montagem do Layout Final
        df_resumo_mec = df_transformado.select([
            pl.col("UF"),
            pl.col(col_A).alias("MERCADORIA"),
            pl.col("_col_C_num").cast(pl.Utf8).str.replace(r"\.", ",").alias("ICM.SAI.BS + ICM.POB.CON"),
            pl.col("_col_D_num").cast(pl.Utf8).str.replace(r"\.", ",").alias("ICM.SAI.EF +  ICM.POB.CON"),
            pl.col(col_Y).alias("TRIB"),
            pl.when(pl.col("_pauta_num") == 0.0).then(pl.lit("N")).otherwise(pl.lit("S")).alias("PRC.PAUTA AT"),
            pl.when(cesta_condicao).then(pl.lit("S")).otherwise(pl.lit("N")).alias("Cesta ?"),
            pl.when(reducao_condicao).then(pl.lit("S")).otherwise(pl.lit("N")).alias("IND. Reducao"),
            pl.col(col_P).alias("ICM.POB.CON"),
            pl.col(col_AT).alias("COMPRADOR"),
            pl.col(col_AU).alias("NOME COMPRADOR"),
            pl.col(col_AV).alias("COMPR.MATRIZ"),
            pl.col(col_AW).alias(" NOME COMPRADOR")
        ])

        # --- EXPORTAÇÃO E LIMPEZA ---
        print(f"   ⚡ Gravando o arquivo final: {NOME_ARQUIVO_SAIDA}...")
        path_temp_saida = os.path.join(PASTA_TEMP, NOME_ARQUIVO_SAIDA)

        df_resumo_mec.write_csv(path_temp_saida, separator=";")

        caminho_destino = os.path.join(PASTA_DESTINO, NOME_ARQUIVO_SAIDA)
        shutil.move(path_temp_saida, caminho_destino)

        # Remover arquivo temporário local SSD
        if os.path.exists(caminho_temp_origem):
            os.remove(caminho_temp_origem)

        # Remover arquivo original do Google Drive
        if os.path.exists(caminho_origem):
            os.remove(caminho_origem)
            print(f"   🗑️ Arquivo original deletado da pasta de origem: {nome_arquivo}")

        print(f"   ✓ Sucesso! Gerado arquivo: {NOME_ARQUIVO_SAIDA}")
        arquivos_processados += 1

print(f"\n🚀 Finalizado com sucesso! Total de arquivos processados: {arquivos_processados}")

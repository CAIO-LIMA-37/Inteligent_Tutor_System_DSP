import os
import time
import pandas as pd
import csv
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Carrega as variáveis do arquivo .env da raiz
load_dotenv(dotenv_path="../src/.env")

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PAINEL DE CONTROLE CENTRAL DO EXPERIMENTO
# 1. TIPO_TESTE: 
# "bruto" (Modelo Direto) OU "tutor" (Teste da Arquitetura RAG do Tutor Inteligente)
TIPO_TESTE = "bruto"

# 2. MODELO_ATIVO: (Descomente 1 por vez)
# MODELO_ATIVO = "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit"
MODELO_ATIVO = "gpt-5"
# MODELO_ATIVO = "deepseek-chat"

# 3. ARQUIVO_ENTRADA
ARQUIVO_ENTRADA = "dados_entrada/questions_PDS_final.txt"

# 4. NOVO: CONFIGURAÇÃO DE COLUNAS A PROCESSAR
# Formato: lista de tuplas (idioma, nome_da_coluna)
# Exemplos:
#   COLUNAS_PROCESSAR = [("PT", "pergunta_pt")]  # Apenas português
#   COLUNAS_PROCESSAR = [("EN", "pergunta_en")]  # Apenas inglês
#   COLUNAS_PROCESSAR = [("PT", "pergunta_pt"), ("EN", "pergunta_en")]  # Ambos (comportamento original)
#   COLUNAS_PROCESSAR = [("ES", "pergunta_es")]  # Se existir coluna em espanhol
COLUNAS_PROCESSAR = [("PT", "pergunta_pt")]  # <-- ALTERE AQUI conforme sua necessidade
# ==============================================================================

class BenchmarkRunner:
    """Gerencia a execução dos testes com o Modelo Bruto (Direto na API/vLLM)."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self._discover_provider(model_name)
        self.client = self._initialize_client()

    def _discover_provider(self, model: str) -> str:
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower: return "openai"
        if "deepseek" in model_lower: return "deepseek"
        return "vllm"

    def _initialize_client(self) -> OpenAI:
        if self.provider == "vllm":
            return OpenAI(api_key="vllm-dummy-key", base_url="http://10.10.80.238:8006/v1")
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        else:
            raise ValueError(f"Provedor {self.provider} não suportado.")

    def _build_streaming_kwargs(self, prompt: str) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        model_lower = self.model_name.lower()
    
        # Lista de modelos que NÃO suportam temperature
        modelos_sem_temperature = ["o1", "o3", "gpt-5"]
    
        if not any(modelo in model_lower for modelo in modelos_sem_temperature):
            kwargs["temperature"] = 0.0
            print(f"[DEBUG] Enviando para API: seed={kwargs.get('seed')}, temp={kwargs.get('temperature')}")
        else:
            print(f"[DEBUG] Enviando para API: seed={kwargs.get('seed')}, temp=NÃO ENVIADO (modelo não suporta)")
    
        return kwargs

    def run_warmup(self):
        logging.info(f"Aquecendo o modelo {self.model_name} (Cold Start)...")
        try:
            kwargs = self._build_streaming_kwargs("Responda apenas 'ok'.")
            kwargs["stream"] = False 
            kwargs.pop("stream_options", None)
            self.client.chat.completions.create(**kwargs)
            logging.info("Aquecimento concluído. Prontos para as requisições.")
        except Exception as e:
            logging.error(f"Falha no aquecimento: {e}")

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        kwargs = self._build_streaming_kwargs(prompt)
        first_token_time, usage_data, full_text = None, None, ""
        start_time = time.time()
        
        try:
            response_stream = self.client.chat.completions.create(**kwargs)
            for chunk in response_stream:
                if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                if chunk.usage is not None:
                    usage_data = chunk.usage
            
            total_time = time.time() - start_time
            ttft = first_token_time - start_time if first_token_time else 0.0
            
            prompt_tokens = usage_data.prompt_tokens if usage_data else 0
            completion_tokens = usage_data.completion_tokens if usage_data else 0
            total_tokens = usage_data.total_tokens if usage_data else 0
            
            itl = (total_time - ttft) if completion_tokens > 1 else 0.0  # ITL agora é tempo total (segundos)
            throughput = completion_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0.0  # Throughput em tokens/segundo
            
        except Exception as e:
            logging.error(f"Erro na inferência bruta: {e}")
            return self._error_dict(e)

        return self._format_dict(full_text, total_time, ttft, itl, throughput, prompt_tokens, completion_tokens, total_tokens)

    def _error_dict(self, error):
        return self._format_dict(f"ERRO: {error}", 0, 0, 0, 0, 0, 0, 0)
        
    def _format_dict(self, text, total_t, ttft, itl, tps, t_in, t_out, t_tot):
        return {
            "Resposta_Gerada": text, "Latencia_Total_s": round(total_t, 4),
            "TTFT_s": round(ttft, 8), "ITL_s": round(itl, 8),
            "Throughput_tps": round(tps, 2), "Tokens_In": t_in,
            "Tokens_Out": t_out, "Tokens_Total": t_tot
        }


class TutorInteligenteRunner(BenchmarkRunner):
    """Gerencia a execução dos testes no Backend do Tutor via Streaming API (RAG)."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session_id = f"benchmark-{int(time.time())}"
        # A flag de streaming que configuramos no FastAPI!
        self.api_url = f"http://127.0.0.1:8000/api/v1/chat/{self.session_id}?stream=true"

    def run_warmup(self):
        logging.info(f"Aquecendo o Tutor Inteligente...")
        try:
            payload = {"query": "ok", "model": self.model_name}
            requests.post(self.api_url, json=payload, timeout=60)
            logging.info("Aquecimento do Tutor concluído.")
        except Exception as e:
            logging.warning(f"Aviso no warmup do Tutor (API está no ar?): {e}")

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        first_token_time = None
        full_text = ""
        payload = {
                    "query": prompt,
                    "model": self.model_name,
                    "seed": 1334,           # ou usar uma variável global SEED
                    "temperature": 0.0
                }

        try:
            # stream=True obriga o requests a capturar o SSE em tempo real
            response = requests.post(self.api_url, json=payload, stream=True, timeout=180)
            response.raise_for_status()
            
            # Iteramos pedaço por pedaço conforme o FastAPI devolve a resposta do vLLM
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_text += chunk
            
            total_time = time.time() - start_time
            ttft = first_token_time - start_time if first_token_time else 0.0
            
            # Heurística acadêmica para estimar tokens quando o metadata de rede não está disponível
            # 1 Token ≈ 4 caracteres
            completion_tokens = max(1, len(full_text) // 4)
            prompt_tokens = max(1, len(prompt) // 4)
            
            itl = (total_time - ttft) if completion_tokens > 1 else 0.0  # ITL agora é tempo total (segundos)
            throughput = completion_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0.0  # Throughput em tokens/segundo

            return self._format_dict(
                full_text, total_time, ttft, itl, throughput, 
                prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
            )
            
        except Exception as e:
            logging.error(f"Erro na comunicação com o Tutor: {e}")
            return self._error_dict(e)


def process_dataset():
    """Lê o dataset, instancia o Runner correto e exporta os resultados dinamicamente."""
    os.makedirs("resultados_brutos", exist_ok=True)
    
    if TIPO_TESTE not in ["bruto", "tutor"]:
        raise ValueError("TIPO_TESTE deve ser 'bruto' ou 'tutor'")
    
    # Validar configuração das colunas
    if not COLUNAS_PROCESSAR:
        raise ValueError("COLUNAS_PROCESSAR não pode estar vazio. Defina pelo menos uma coluna.")
    
    # Verificar se as colunas existem no dataset
    df_check = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8', nrows=1)
    for idioma, coluna in COLUNAS_PROCESSAR:
        if coluna not in df_check.columns:
            raise ValueError(f"Coluna '{coluna}' não encontrada no arquivo. Colunas disponíveis: {list(df_check.columns)}")
    
    safe_model_name = MODELO_ATIVO.replace("/", "_")
    # Criar nome do arquivo com as colunas usadas
    colunas_sufixo = "_".join([col for _, col in COLUNAS_PROCESSAR])
    output_csv = f"resultados_brutos/geracao_{TIPO_TESTE}_{safe_model_name}_{colunas_sufixo}.csv"
    
    logging.info(f"TESTE: {TIPO_TESTE.upper()} | MODELO: {MODELO_ATIVO}")
    logging.info(f"COLUNAS A PROCESSAR: {COLUNAS_PROCESSAR}")
    
    df = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    
    # Instancia dinamicamente o orquestrador correto com base no Painel de Controle
    runner = BenchmarkRunner(MODELO_ATIVO) if TIPO_TESTE == "bruto" else TutorInteligenteRunner(MODELO_ATIVO)
    runner.run_warmup()
    
    # Adicionada a coluna "Interferencia_Tutor" para os gráficos comparativos do Pandas
    colunas = ["Idioma", "Interferencia_Tutor", "Capitulo", "Num_Questao", "Pergunta", "Modelo", 
               "Resposta_Gerada", "Latencia_Total_s", "TTFT_s", "ITL_s", 
               "Throughput_tps", "Tokens_In", "Tokens_Out", "Tokens_Total"]
    pd.DataFrame(columns=colunas).to_csv(output_csv, index=False, encoding='utf-8')
    
    status_tutor = "NÃO" if TIPO_TESTE == "bruto" else "SIM"
    
    # Processar cada bloco de coluna configurado
    for idioma, coluna_pergunta in COLUNAS_PROCESSAR:
        logging.info(f"=== INICIANDO BLOCO: {idioma} (Coluna: {coluna_pergunta}) ===")
        
        for index, row in df.iterrows():
            pergunta = row[coluna_pergunta]
            
            # Verificar se a pergunta é válida (não NaN e não vazia)
            if pd.isna(pergunta) or str(pergunta).strip() == "":
                logging.warning(f"[{idioma}] Q{index+1} - Pergunta vazia ou inválida. Pulando...")
                continue
                
            logging.info(f"[{idioma}] Processando Q{index+1}/{len(df)} | Cap: {row['capitulo']}")
            
            m = runner.generate_response(pergunta)
            linha = {
                "Idioma": idioma,
                "Interferencia_Tutor": status_tutor,
                "Capitulo": row['capitulo'],
                "Num_Questao": row['numero_questao'],
                "Pergunta": pergunta,
                "Modelo": MODELO_ATIVO,
                "Resposta_Gerada": m["Resposta_Gerada"],
                "Latencia_Total_s": m["Latencia_Total_s"],
                "TTFT_s": m["TTFT_s"],
                "ITL_s": m["ITL_s"],
                "Throughput_tps": m["Throughput_tps"],
                "Tokens_In": m["Tokens_In"],
                "Tokens_Out": m["Tokens_Out"],
                "Tokens_Total": m["Tokens_Total"]
            }
            
            pd.DataFrame([linha]).to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
            time.sleep(0.5)
    
    logging.info(f"Benchmark concluído com sucesso! Salvo em: {output_csv}")

# Função utilitária para listar colunas disponíveis no dataset
def listar_colunas_disponiveis():
    """Helper function para ver quais colunas existem no dataset"""
    df = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8', nrows=1)
    print("Colunas disponíveis no dataset:")
    for col in df.columns:
        print(f"  - {col}")

if __name__ == "__main__":
    # Opcional: descomente para ver colunas disponíveis antes de rodar
    # listar_colunas_disponiveis()
    process_dataset()
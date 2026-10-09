import os
import subprocess
import sys
import time

# Nomes exatos dos seus arquivos (conforme seus uploads)
GENERATOR_SCRIPT = "Generator_NEW.py"
NOTEBOOK_SL = "SL_FINAL (1).ipynb"

def run_command(command, description):
    print(f"\n>>> ⏳ {description}...")
    start = time.time()
    try:
        # Executa o comando e aguarda o término
        subprocess.check_call(command, shell=True)
        print(f"   ✅ Concluído em {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ ERRO FATAL ao executar: {description}")
        sys.exit(1)

def run_pipeline():
    print("="*60)
    print("🚀 INICIANDO PIPELINE DE AUTOMATIZAÇÃO (LOCAC)")
    print("="*60)
    
    # Garante que estamos na pasta do projeto
    if not os.path.exists(GENERATOR_SCRIPT):
        print(f"❌ Erro: Não foi possível encontrar {GENERATOR_SCRIPT}. Execute de dentro da pasta 'project' ou ajuste os caminhos.")
        sys.exit(1)

    # 1. Gerar Dados (Gêmeo Digital)
    run_command(f"{sys.executable} {GENERATOR_SCRIPT}", "1. Gerando Dados Sintéticos e Scalers")

    # 2. Treinar SL (Baseline)
    # Usa nbconvert para executar o notebook como se fosse um script
    run_command(
        f"{sys.executable} -m jupyter nbconvert --to notebook --execute --inplace \"{NOTEBOOK_SL}\"", 
        "2. Treinando Modelo Supervisionado (SL)"
    )

    # 2b. Treinar e exportar o bandido contextual (referência do RL)
    run_command(f"{sys.executable} treinar_bandido.py", "2b. Treinando Bandido Contextual (LightGBM)")

    # 2c. Treinar e exportar o behavior cloning (imitação do histórico)
    run_command(f"{sys.executable} treinar_bc.py", "2c. Treinando Behavior Cloning (BC)")

    # 3. Treinar o RL (CQL) pela versão canônica (politicas.CQL); os notebooks
    #    de RL ficam só como documentação da versão anterior
    run_command(f"{sys.executable} treinar_cql.py", "3. Treinando RL (CQL) - venda única e assinatura")

    # A comparação com os baselines (várias sementes, retreina o CQL) é um passo
    # separado e demorado: python avaliar_politicas.py (docs/protocolo_avaliacao.md)

    print("\n" + "="*60)
    print("🎉 PIPELINE CONCLUÍDO: Todos os modelos foram treinados e salvos.")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
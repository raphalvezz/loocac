import os
import uvicorn
import sys
import subprocess

# Caminhos
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(ROOT_DIR, "project")
REQUIRED_MODELS = [
    "modelo_rl_final.pt",
    "modelo_rl_assinatura.pt",
    "sl_profit_regressor_model.joblib",
    "scaler_estado.joblib" # Verifica um scaler também para garantir
]

def check_artifacts():
    """Verifica se os modelos necessários existem."""
    missing = []
    for file in REQUIRED_MODELS:
        if not os.path.exists(os.path.join(PROJECT_DIR, file)):
            missing.append(file)
    return missing

def main():
    # 1. Entra na pasta do projeto
    if not os.path.isdir(PROJECT_DIR):
        print(f"❌ Erro: Diretório 'project' não encontrado em: {PROJECT_DIR}")
        sys.exit(1)
    
    os.chdir(PROJECT_DIR)
    print(f"📂 Diretório de trabalho: {os.getcwd()}")

    # 2. Verifica integridade do sistema
    missing_files = check_artifacts()
    
    if missing_files:
        print("\n⚠️  AVISO: Arquivos de IA essenciais não encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n⚙️  Iniciando AUTO-CONFIGURAÇÃO (Isso acontece apenas na primeira vez)...")
        print("   O sistema irá gerar dados e treinar os modelos agora.")
        
        # Chama o pipeline que criamos no Passo 1
        try:
            subprocess.check_call([sys.executable, "train_pipeline.py"])
        except subprocess.CalledProcessError:
            print("\n❌ Falha na auto-configuração. Verifique os logs acima.")
            sys.exit(1)
    else:
        print("\n✅ Sistema íntegro: Todos os modelos e dados estão prontos.")

    # 3. Inicia o Servidor
    print("\n🌐 Iniciando Servidor Backend (FastAPI)...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
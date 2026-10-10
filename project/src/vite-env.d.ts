/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_LOCAC_API_URL?: string;  // padrão: http://127.0.0.1:8000
  readonly VITE_LOCAC_API_KEY?: string;  // a mesma LOCAC_API_KEY da API
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

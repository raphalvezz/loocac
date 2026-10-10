export interface Location {
  name: string;
  searchVolume: number;
  cpc: number;
  competition: number;
}

export interface CampaignSimulation {
  keyword: string;
  budget: number;
  targetRevenue: number;
  pricingModel: 'subscription' | 'fixed';
  countries: Location[];
  states: Location[];
  cities: Location[];
  recommendations: Recomendacao[];
}

export type MetodoPreco = 'rl' | 'sl' | 'bandido' | 'bc';

// Resposta de POST /recomendar e de cada item de POST /comparar (main.py)
export interface Recomendacao {
  metodo: MetodoPreco;
  cobranca: 'venda_unica' | 'assinatura';
  preco_recomendado: number;
  preco_bruto: number;                    // saída do método antes da regra de faixa (KBS)
  kbs_applied: boolean;
  lucro_previsto_metodo: number | null;   // modelo de lucro do próprio método (BC não tem)
  lucro_estimado_sl: number | null;       // avaliação comum: SL no preço final
  var_5_percent: number | null;           // avaliação comum: quantis do crítico do RL
  cvar_5_percent: number | null;
  latencia_ms: number;
}

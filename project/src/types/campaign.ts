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
  recommendations: PricingRecommendation[];
}

export interface PricingRecommendation {
  type: 'subscription' | 'fixed';
  amount: number;
  rawAmount: number;       // preço do RL antes da regra de faixa (KBS)
  kbsApplied: boolean;
  estimatedProfit: number; // lucro previsto pelo SL
  roi: number;             // lucro / orçamento, em %
  var5: number;
  cvar5: number;
  locations: string[];
}
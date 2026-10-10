import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { DollarSign } from 'lucide-react';
import { MetodoPreco, Recomendacao } from '../types/campaign';

// Configuração em .env.local (ver .env.example): URL da API e a mesma chave da LOCAC_API_KEY
const api = axios.create({
  baseURL: import.meta.env.VITE_LOCAC_API_URL ?? 'http://127.0.0.1:8000',
  headers: { 'X-API-Key': import.meta.env.VITE_LOCAC_API_KEY ?? '' },
});

const METODOS: { valor: MetodoPreco; rotulo: string }[] = [
  { valor: 'rl', rotulo: 'RL (CQL)' },
  { valor: 'bandido', rotulo: 'Bandido contextual' },
  { valor: 'sl', rotulo: 'Supervisionado (SL)' },
  { valor: 'bc', rotulo: 'Imitação (BC)' },
];
const ROTULO: Record<MetodoPreco, string> = Object.fromEntries(METODOS.map((m) => [m.valor, m.rotulo])) as Record<MetodoPreco, string>;

const mensagemDeErro = (erro: unknown) => {
  if (axios.isAxiosError(erro)) {
    if (erro.response?.status === 401) return 'Chave de API ausente ou inválida: confira VITE_LOCAC_API_KEY.';
    const detalhe = erro.response?.data?.detail;
    if (typeof detalhe === 'string') return detalhe;
    if (!erro.response) return 'API fora do ar ou bloqueada por CORS.';
  }
  return 'Erro ao simular a campanha. Verifique o backend.';
};

const CampaignSimulatorPage = () => {
  // 1. Configuração de Mercado (Gêmeo Digital)
  const [marketConfig, setMarketConfig] = useState({
    lowMin: 10, lowMax: 97,
    highMin: 497, highMax: 5000,
    budgetMin: 500, budgetMax: 20000
  });
  const [isRetraining, setIsRetraining] = useState(false);
  const [marketMessage, setMarketMessage] = useState<string | null>(null);

  // 2. Formulário (estado da campanha + método)
  const [formData, setFormData] = useState({
    region: 'North America',
    platform: 'Instagram',
    product_tier: 'Low Ticket',
    budget: 1000,
    pricingModel: 'fixed',
    metodo: 'comparar' as MetodoPreco | 'comparar',
  });
  const [showResults, setShowResults] = useState(false);

  // Salva as faixas no backend (config_market.json) e dispara o re-treino em background
  const handleUpdateMarket = async () => {
    setIsRetraining(true);
    setMarketMessage(null);
    try {
      const response = await api.post('/configure_market', marketConfig);
      setMarketMessage(response.data.message);
    } catch (erro) {
      setMarketMessage(mensagemDeErro(erro));
    } finally {
      setIsRetraining(false);
    }
  };

  const marketFields: { key: keyof typeof marketConfig; label: string }[] = [
    { key: 'lowMin', label: 'Low Ticket min' },
    { key: 'lowMax', label: 'Low Ticket max' },
    { key: 'highMin', label: 'High Ticket min' },
    { key: 'highMax', label: 'High Ticket max' },
    { key: 'budgetMin', label: 'Budget min' },
    { key: 'budgetMax', label: 'Budget max' },
  ];

  // 3. Consulta à API: um método (/recomendar) ou todos (/comparar)
  const { data: recomendacoes, isFetching, error, refetch } = useQuery({
    queryKey: ['campaignSimulation', formData],
    queryFn: async (): Promise<Recomendacao[]> => {
      // Nomes dos campos conforme o CampaignInput da API (main.py)
      const payload = {
        Regiao: formData.region,
        Plataforma: formData.platform,
        Tier: formData.product_tier,
        Orcamento: Number(formData.budget),
      };
      const cobranca = formData.pricingModel === 'subscription' ? 'assinatura' : 'venda_unica';
      if (formData.metodo === 'comparar') {
        return (await api.post<Recomendacao[]>('/comparar', payload, { params: { cobranca } })).data;
      }
      const params = { metodo: formData.metodo, cobranca };
      return [(await api.post<Recomendacao>('/recomendar', payload, { params })).data];
    },
    enabled: false,
    retry: false,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(true);
    refetch();
  };

  const formatCurrency = (amount: number | null) =>
    amount === null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);

  // Na comparação, destaca o maior lucro pela avaliação comum (SL), não pelo modelo de cada
  // método; empates (o SL é por árvores, preços próximos podem cair na mesma folha) destacam todos
  const maiorLucro = recomendacoes && recomendacoes.length > 1
    ? Math.max(...recomendacoes.map((r) => r.lucro_estimado_sl ?? -Infinity))
    : null;
  const ehMelhor = (rec: Recomendacao) => maiorLucro !== null && rec.lucro_estimado_sl === maiorLucro;

  const selectClass = "mt-1 block w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white";

  return (
    <div className="pb-16 lg:pb-0">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Campaign Simulator</h1>

          {/* Painel de Configuração de Mercado */}
          <div className="mb-8 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
             <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Gêmeo Digital: faixas de mercado</h2>
             <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
               {marketFields.map(({ key, label }) => (
                 <label key={key} className="block text-xs text-gray-600 dark:text-gray-300">
                   {label}
                   <input type="number" value={marketConfig[key]} onChange={(e) => setMarketConfig({ ...marketConfig, [key]: Number(e.target.value) })} className="mt-1 block w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white" />
                 </label>
               ))}
             </div>
             <button onClick={handleUpdateMarket} disabled={isRetraining} className="w-full py-2 bg-secondary-600 text-white rounded hover:bg-secondary-700 text-sm font-medium disabled:opacity-50">
                {isRetraining ? "Re-treinando..." : "Atualizar Gêmeo Digital & Re-treinar"}
             </button>
             {marketMessage && <p className="mt-2 text-xs text-gray-600 dark:text-gray-300">{marketMessage}</p>}
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Monthly Budget</label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><DollarSign className="h-5 w-5 text-gray-400" /></div>
                  <input type="number" min={1} value={formData.budget} onChange={(e) => setFormData({ ...formData, budget: Number(e.target.value) })} className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white" required />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Product Tier</label>
                <select value={formData.product_tier} onChange={(e) => setFormData({ ...formData, product_tier: e.target.value })} className={selectClass}>
                  <option value="Low Ticket">Low Ticket</option>
                  <option value="High Ticket">High Ticket</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Pricing Model</label>
                <select value={formData.pricingModel} onChange={(e) => setFormData({ ...formData, pricingModel: e.target.value })} className={selectClass}>
                  <option value="fixed">Fixed Price</option>
                  <option value="subscription">Subscription</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Method</label>
                <select value={formData.metodo} onChange={(e) => setFormData({ ...formData, metodo: e.target.value as MetodoPreco | 'comparar' })} className={selectClass}>
                  <option value="comparar">Comparar todos</option>
                  {METODOS.map((m) => <option key={m.valor} value={m.valor}>{m.rotulo}</option>)}
                </select>
              </div>

              {/* Mesmas regiões e plataformas do simulador (simulador.py) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Region</label>
                <select value={formData.region} onChange={(e) => setFormData({ ...formData, region: e.target.value })} className={selectClass}>
                  <option value="North America">North America</option>
                  <option value="Europe">Europe</option>
                  <option value="Asia">Asia</option>
                  <option value="South America">South America</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Platform</label>
                <select value={formData.platform} onChange={(e) => setFormData({ ...formData, platform: e.target.value })} className={selectClass}>
                  <option value="Instagram">Instagram</option>
                  <option value="Facebook">Facebook</option>
                  <option value="LinkedIn">LinkedIn</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end">
              <button type="submit" disabled={isFetching} className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50">
                {isFetching ? "Simulating..." : "Run Simulation"}
              </button>
            </div>
          </form>

          {showResults && !isFetching && recomendacoes && (
            <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                {recomendacoes.length > 1 ? 'Comparação de métodos' : 'AI Recommendation'}
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
                Todos os preços passam pela mesma regra de faixa (KBS). Lucro (SL), ROI e risco são calculados pelo mesmo avaliador para todos os métodos.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {recomendacoes.map((rec) => (
                  <div key={rec.metodo} className={`bg-white dark:bg-gray-700 rounded-lg shadow-sm border p-5 ${ehMelhor(rec) ? 'border-primary-500 ring-1 ring-primary-500' : 'border-gray-200 dark:border-gray-600'}`}>
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">{ROTULO[rec.metodo]}</p>
                        {ehMelhor(rec) && <p className="text-xs text-primary-600 dark:text-primary-400">Maior lucro previsto (SL)</p>}
                        <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{formatCurrency(rec.preco_recomendado)}</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">{rec.cobranca === 'assinatura' ? 'per month' : 'one-time'}</p>
                        {rec.kbs_applied && (
                          <p className="mt-1 text-xs text-amber-600">Ajustado à faixa do tier (método sugeriu {formatCurrency(rec.preco_bruto)})</p>
                        )}
                      </div>
                      <ul className="space-y-1.5 text-sm">
                        {[
                          ['Lucro (método)', formatCurrency(rec.lucro_previsto_metodo)],
                          ['Lucro (SL)', formatCurrency(rec.lucro_estimado_sl)],
                          ['ROI (SL)', rec.lucro_estimado_sl === null ? '—' : `${((rec.lucro_estimado_sl / formData.budget) * 100).toFixed(1)}%`],
                          ['VaR 5%', formatCurrency(rec.var_5_percent)],
                          ['CVaR 5%', formatCurrency(rec.cvar_5_percent)],
                        ].map(([rotulo, valor]) => (
                          <li key={rotulo} className="flex justify-between gap-3">
                            <span className="text-gray-500 dark:text-gray-400 whitespace-nowrap">{rotulo}</span>
                            <span className="font-medium text-gray-900 dark:text-white whitespace-nowrap tabular-nums">{valor}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-50 p-4 rounded text-red-700">{mensagemDeErro(error)}</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CampaignSimulatorPage;

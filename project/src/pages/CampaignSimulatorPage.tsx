import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { DollarSign, Calculator, Settings } from 'lucide-react'; // Removidos ícones não usados
import { CampaignSimulation, PricingRecommendation } from '../types/campaign';

const CampaignSimulatorPage = () => {
  // 1. Estado da Configuração de Mercado (Gêmeo Digital)
  const [marketConfig, setMarketConfig] = useState({
    lowMin: 10, lowMax: 97,
    highMin: 497, highMax: 5000,
    budgetMin: 500, budgetMax: 20000
  });
  const [isRetraining, setIsRetraining] = useState(false);

  // 2. Estado do Formulário (Inputs Reais da IA)
  const [formData, setFormData] = useState({
    region: 'North America',
    platform: 'Instagram',
    product_tier: 'Low Ticket',
    budget: 1000,
    age: '25-34',      // Adicionado (Essencial para a IA)
    gender: 'Female',  // Adicionado (Essencial para a IA)
    content: 'Image',  // Adicionado (Essencial para a IA)
    pricingModel: 'fixed'
  });
  
  const [showResults, setShowResults] = useState(false);

  // ... (Função handleUpdateMarket mantém-se igual) ...
  const handleUpdateMarket = async () => { /* ... código anterior ... */ };

  // 3. Query para API
  const { data: simulation, isLoading, error, refetch } = useQuery({
    queryKey: ['campaignSimulation', formData],
    queryFn: async () => {
      const endpoint = formData.pricingModel === 'fixed' ? '/recommend_price' : '/recommend_subscription_price';
      const response = await axios.post(`http://127.0.0.1:8000${endpoint}`, formData);
      const rlData = response.data; 

      const recommendation: PricingRecommendation = {
        type: formData.pricingModel === 'subscription' ? 'subscription' : 'fixed',
        amount: rlData.preco_recomendado,
        estimatedRevenue: (formData.budget * ((rlData.lucro_estimado_sl || 1.5))) + formData.budget, // Estimativa baseada no SL
        roi: ((rlData.lucro_estimado_sl || 0) / formData.budget) * 100,
        coverage: 0.85,
        locations: [formData.region]
      };
      
      // Simplificado: Removemos os arrays vazios de location
      return { recommendations: [recommendation] };
    },
    enabled: false,
  }); 

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(true);
    refetch();
  };

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);

  return (
    <div className="pb-16 lg:pb-0">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Campaign Simulator</h1>

          {/* Painel de Configuração de Mercado */}
          <div className="mb-8 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
             {/* ... (Código do Painel de Configuração mantém-se igual) ... */}
             <button onClick={handleUpdateMarket} disabled={isRetraining} className="w-full py-2 bg-secondary-600 text-white rounded hover:bg-secondary-700 text-sm font-medium disabled:opacity-50">
                {isRetraining ? "Re-treinando..." : "Atualizar Gêmeo Digital & Re-treinar"}
             </button>
          </div>

          {/* Formulário Limpo */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Orçamento */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Monthly Budget</label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><DollarSign className="h-5 w-5 text-gray-400" /></div>
                  <input type="number" value={formData.budget} onChange={(e) => setFormData({ ...formData, budget: Number(e.target.value) })} className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white" required />
                </div>
              </div>

              {/* Tier do Produto */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Product Tier</label>
                <select value={formData.product_tier} onChange={(e) => setFormData({ ...formData, product_tier: e.target.value })} className="mt-1 block w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option value="Low Ticket">Low Ticket</option>
                  <option value="High Ticket">High Ticket</option>
                </select>
              </div>

              {/* Modelo de Preço */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Pricing Model</label>
                <select value={formData.pricingModel} onChange={(e) => setFormData({ ...formData, pricingModel: e.target.value })} className="mt-1 block w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option value="fixed">Fixed Price</option>
                  <option value="subscription">Subscription</option>
                </select>
              </div>

              {/* Região */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Region</label>
                <select value={formData.region} onChange={(e) => setFormData({ ...formData, region: e.target.value })} className="mt-1 block w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option value="North America">North America</option>
                  <option value="Europe">Europe</option>
                  <option value="Asia">Asia</option>
                  <option value="South America">South America</option>
                  <option value="Africa">Africa</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end">
              <button type="submit" disabled={isLoading} className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50">
                {isLoading ? "Simulating..." : "Run Simulation"}
              </button>
            </div>
          </form>

          {/* Resultados Limpos */}
          {showResults && !isLoading && simulation && (
            <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">AI Recommendation</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {simulation.recommendations.map((rec, index) => (
                    <div key={index} className="bg-white dark:bg-gray-700 rounded-lg shadow-sm border border-gray-200 dark:border-gray-600 p-6">
                      <div className="space-y-4">
                        <div>
                          <p className="text-3xl font-bold text-gray-900 dark:text-white">{formatCurrency(rec.amount)}</p>
                          <p className="text-sm text-gray-500 dark:text-gray-400">{rec.type === 'subscription' ? 'per month' : 'one-time'}</p>
                        </div>
                        <ul className="space-y-2">
                          <li className="flex justify-between text-sm"><span className="text-gray-500">Est. Profit (SL)</span><span className="font-medium">{formatCurrency(rec.estimatedRevenue - formData.budget)}</span></li>
                          <li className="flex justify-between text-sm"><span className="text-gray-500">ROI</span><span className="font-medium">{rec.roi.toFixed(1)}%</span></li>
                        </ul>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-50 p-4 rounded text-red-700">Error simulating campaign. Check backend.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CampaignSimulatorPage;
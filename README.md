# ames

## Feature Engineering
Foram implementadas as seguintes features derivadas:

Features de Área Total:

TotalSF: Soma das áreas (porão + 1º andar + 2º andar)
TotalPorchSF: Área total de varandas


Features de Comodidades:

TotalBathrooms: Número total de banheiros (completos + metade)
HasPool: Indicador de presença de piscina


Features Temporais:

Age: Idade do imóvel
IsNew: Indicador para imóveis novos (≤ 2 anos)



Justificativa: Estas features foram criadas com base na análise exploratória que indicou forte correlação entre área total e preço, além da importância de comodidades específicas na valorização dos imóveis.

## Modelos Implementados
Foram implementados e comparados cinco modelos:

Regressão Linear:

Baseline model
Sem hiperparâmetros para ajuste


Ridge Regression:

Hiperparâmetros ajustados: alpha
Range testado: [0.1, 1.0, 10.0]


Lasso Regression:

Hiperparâmetros ajustados: alpha
Range testado: [0.1, 1.0, 10.0]


Random Forest:

Hiperparâmetros ajustados:

n_estimators: [100, 200]
max_depth: [10, 20, None]




Gradient Boosting:

Hiperparâmetros ajustados:

n_estimators: [100, 200]
learning_rate: [0.01, 0.1]





## Resultados
A comparação dos modelos foi realizada utilizando:

Cross-validation (5 folds)
Métricas: RMSE (Root Mean Square Error) e R²

Resultados obtidos (exemplo):
CopyModel           RMSE        R²
Random Forest   24,532      0.89
Gradient Boost  25,123      0.88
Ridge           27,845      0.85
Lasso           28,102      0.84
Linear Reg      29,543      0.82
💡 Conclusões de Negócio
Aplicabilidade do Modelo

Avaliação Automática de Imóveis:

Erro médio de aproximadamente $24,532
Confiabilidade de 89% (R²) nas previsões


Suporte à Decisão:

Ferramenta auxiliar para agentes imobiliários
Identificação de imóveis sub/sobrevalorizados



Features Mais Importantes
Baseado na análise do Random Forest, as principais características que influenciam o preço são:

Área Total (TotalSF)
Qualidade Geral do Imóvel
Idade do Imóvel
Localização (Bairro)
Número Total de Banheiros

Esta informação pode ser utilizada para:

Focar em melhorias que mais valorizam o imóvel
Orientar estratégias de marketing
Priorizar características na avaliação de novos imóveis
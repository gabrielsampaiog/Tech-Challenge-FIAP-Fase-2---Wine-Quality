# Classificação da Qualidade de Vinhos com ML

Pipeline de Machine Learning construído para prever a qualidade de vinhos com base em suas características físico-químicas.

## Estrutura de Pastas
* **`data/`**: Base de dados original (`WineQT.csv`).
* **`notebooks/`**: Jupyter Notebook com o código completo.
* **`results/`**: Gráficos de análise, correlação e matrizes de confusão.
* **`src/`** e **`requirements.txt`**: Scripts auxiliares e bibliotecas.

---

## Etapas do Desafio
1. **Compreensão do Problema:** Criação da variável alvo binária (`quality_binary`), onde nota >= 7 é alta qualidade (Classe 1) e < 7 é baixa/média qualidade (Classe 0).
2. **Análise Exploratória (EDA):** Análise de balanceamento de classes, boxplots de distribuição e matriz de correlação (salvos em `results/`).
3. **Pré-processamento:** Limpeza e aplicação de `StandardScaler` para padronizar as escalas das variáveis.
4. **Modelos:** Treinamento do modelo baseline (*Regressão Logística*) e do modelo avançado (*Random Forest*).
5. **Avaliação:** Comparação via matrizes de confusão e métricas de classificação (*Acurácia*, *Precision*, *Recall* e *F1-Score*).
6. **Resultados:** O modelo identificou o *Teor Alcoólico* e a *Acidez Volátil* como os principais fatores de impacto na qualidade do vinho.

---

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt

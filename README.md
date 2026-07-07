# Rainfall Prediction

Projeto de ciencia de dados para prever se chovera amanha usando dados meteorologicos historicos da Australia.

Este repositorio mostra a refatoracao de um projeto antigo de Deep Learning para um fluxo mais profissional de Machine Learning: prevencao de vazamento de dados, comparacao de modelos, metricas adequadas, validacao temporal, relatorios e uma pequena demo interativa.

## Por que este projeto importa

Prever chuva com antecedencia pode apoiar decisoes em logistica, agricultura, construcao civil, transporte e gestao de risco climatico.

Mais importante do que buscar uma acuracia muito alta e construir uma avaliacao honesta: o modelo deve usar apenas informacoes que estariam disponiveis no momento real da previsao.

## Principal cuidado tecnico

O dataset original contem a coluna `RISK_MM`, que representa a quantidade de chuva do dia seguinte. Essa variavel esta diretamente ligada ao alvo `RainTomorrow` e causaria vazamento de dados.

Por isso, `RISK_MM` e removida antes do treino. Sem essa correcao, o modelo poderia parecer excelente, mas nao generalizaria para um cenario real.

## Estrutura

```text
rainfall-prediction/
|-- app.py
|-- data/
|   `-- database
|-- notebooks/
|   `-- 01_rainfall_prediction_story.ipynb
|-- reports/
|   |-- metrics.csv
|   |-- temporal_metrics.csv
|   |-- evaluation.json
|   `-- figures/
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- evaluate.py
|   |-- features.py
|   |-- models.py
|   `-- train.py
|-- tests/
|-- requirements.txt
`-- README.md
```

O arquivo `models/rainfall_model.joblib` e gerado localmente pelo treino e nao e versionado por ser pesado.

## Modelos comparados

O treinamento compara tres abordagens:

- `DummyClassifier`: baseline minimo.
- `Logistic Regression`: modelo linear interpretavel.
- `Random Forest`: modelo nao linear para capturar interacoes entre variaveis meteorologicas.

## Metricas

O projeto usa metricas mais uteis do que acuracia isolada:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- matriz de confusao
- curva ROC
- curva Precision-Recall
- importancia de variaveis

Para previsao de chuva, `recall` e especialmente importante porque mede a capacidade de detectar dias em que realmente vai chover.

## Resultados atuais

Avaliacao com divisao estratificada de treino e teste:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.834 | 0.612 | 0.709 | 0.657 | 0.883 |
| Logistic Regression | 0.794 | 0.527 | 0.777 | 0.628 | 0.873 |
| Dummy baseline | 0.776 | 0.000 | 0.000 | 0.000 | 0.500 |

Validacao temporal, treinando em dados mais antigos e testando em dados mais recentes:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.825 | 0.593 | 0.683 | 0.635 | 0.866 |
| Logistic Regression | 0.788 | 0.517 | 0.752 | 0.613 | 0.858 |
| Dummy baseline | 0.777 | 0.000 | 0.000 | 0.000 | 0.500 |

Esses resultados sao mais realistas do que a acuracia anterior acima de 98%, porque removem a variavel de vazamento `RISK_MM`.

## Como rodar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Treine e avalie os modelos:

```bash
python -m src.train
```

Rode os testes:

```bash
python -m unittest discover
```

Abra a demo:

```bash
streamlit run app.py
```

Abra a apresentacao em notebook:

```bash
jupyter notebook notebooks/01_rainfall_prediction_story.ipynb
```

## Saidas geradas

Depois do treino, o projeto cria:

- `reports/metrics.csv`: comparacao dos modelos.
- `reports/temporal_metrics.csv`: comparacao em validacao temporal.
- `reports/evaluation.json`: metricas detalhadas.
- `reports/figures/confusion_matrix.png`: matriz de confusao.
- `reports/figures/roc_curve.png`: curva ROC.
- `reports/figures/precision_recall_curve.png`: curva Precision-Recall.
- `reports/figures/feature_importance.png`: variaveis mais importantes.
- `models/rainfall_model.joblib`: melhor modelo treinado.

## Proximos passos

- Adicionar SHAP para explicabilidade local das previsoes.
- Criar testes adicionais para o pipeline de treino.
- Publicar a demo em Streamlit Community Cloud.

## Licenca

Este projeto esta licenciado sob a licenca MIT.

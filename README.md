# Loan Approval Prediction

Projeto de classificação para previsão de aprovação de empréstimos, utilizando métodos clássicos de Machine Learning (KNN, Árvore de Decisão, Naive Bayes e SVM). Feito para o MVP de Qualidade de Software, Segurança e Sistemas Inteligentes.

## Instalação
1. Crie uma venv
2. Após ativar a venv, dentro do loan_app, utilize:
```bash
pip install -r requirements.txt
```

## Rodando a Aplicação

```bash
cd loan_app
python app.py
```
Acesse em `http://localhost:5000`.

## Testes

```bash
cd loan_app
pytest tests/test_model.py -v
```
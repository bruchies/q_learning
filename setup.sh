#!/bin/bash

echo "🔧 Criando ambiente virtual .venv..."
python3 -m venv .venv

echo "✅ Ambiente criado."

echo "⚡ Ativando ambiente virtual..."
source .venv/bin/activate

echo "⬇️ Instalando pygame..."
pip install --upgrade pip
pip install pygame

echo "📝 Gerando requirements.txt..."
pip freeze > requirements.txt

echo "✅ Tudo pronto!"
echo "ℹ️ Para ativar depois, use: source .venv/bin/activate"

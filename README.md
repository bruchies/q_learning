
# Relatório do Projeto: Agente com Q-Learning

Este relatório explica o funcionamento do código implementado para o projeto da disciplina **Fundamentos de Inteligência Artificial**, cujo objetivo é treinar um agente a encontrar o melhor caminho em um ambiente 2D usando o algoritmo de **Q-Learning**.

---

## 1. Estrutura do Projeto

O código foi modularizado da seguinte forma:

```
q_learning/
├── src/
│   ├── main.py          # Executa o loop principal com renderização
│   ├── agent.py         # Implementa o algoritmo Q-Learning
│   ├── gui.py           # Cuida da interface gráfica com Pygame
│   └── config.py        # Contém constantes e parâmetros
├── requirements.txt     # Lista de dependências
└── README.md            # Relatório explicativo (este arquivo)
```

---

## 2. Configurações Iniciais (config.py)

Este arquivo define:

- `ROWS`, `COLS`: dimensões do ambiente (10x12)
- `ACTIONS`: lista de ações possíveis (`up`, `down`, `left`, `right`)
- `EPSILON`, `GAMMA`: parâmetros do algoritmo Q-Learning
- `reward_matrix`: define os valores de recompensa do mapa
- `initial_state`, `terminal_state`: estado inicial e objetivo final

---

## 3. Algoritmo Q-Learning (agent.py)

### Principais funções:

- `is_valid(state)`: verifica se o estado é válido (não é parede).
- `get_next_state(state, action)`: retorna o próximo estado dado uma ação.
- `choose_action(state)`: seleciona a ação com melhor Q-value ou uma aleatória (ε-greedy).
- `update_q(state, action, reward, next_state)`: aplica a equação de Bellman para atualizar a Tabela Q.

A Tabela Q é uma matriz 3D: `Q[linha][coluna][ação]`, representando o valor esperado de executar uma ação em determinado estado.

---

## 4. Interface Gráfica (gui.py)

### Visualizações:

- **Setas**: exibe a melhor ação aprendida para cada estado.
- **Mapa de Calor**: representa frequência de visitas por um gradiente roxo → rosa.
- **Tabela Q**: aparece ao lado do mapa, mostrando os melhores Q-values de cada estado válido.

### Desenho:

- Cada célula do mapa é desenhada com uma cor baseada em seu tipo (parede, obstáculo, caminho, objetivo).
- Um círculo azul representa o agente.
- A pontuação atual é exibida no topo.

---

## 5. Execução Principal (main.py)

### Lógica:

- Inicializa o ambiente com `pygame`
- Treina o agente por 1000 episódios.
- A cada passo, o agente escolhe a ação com melhor Q-value e se move.
- Quando chega ao estado final, um novo episódio começa.
- Pressionar `ESPAÇO` alterna entre visualização por setas e por calor.

---

## 6. Como Executar

### Requisitos:

- Python 3.12
- pygame

### Instalação:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

---

## 7. Observações

- O agente evita ciclos escolhendo aleatoriamente entre ações com valores Q iguais.
- O ambiente é irregular: apenas colunas centrais das últimas 5 linhas são válidas.
- As cores da visualização são personalizadas para facilitar a leitura.

---

## 8. Créditos

**Bruna Chies**  
**Jhonatan Martins**  
Curso: Ciência da Computação  
Disciplina: Fundamentos de Inteligência Artificial  
Junho/2025

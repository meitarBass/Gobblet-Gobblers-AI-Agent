# Gobblet Gobblers AI Agent  

An implementation of the strategy board game **Gobblet Gobblers** along with an AI agent capable of playing the game intelligently.  
This project provides both the complete game logic and an AI agent that can play against human players or other agents.  

---

## 🎯 Features  

- Full implementation of **Gobblet Gobblers** rules and mechanics  
- AI agent that can make decisions and play the game  
- Separation of game logic and AI logic for modularity  
- Playable via command line interface  
- Includes official rules and background material  

---

## 📂 Repository Structure  

```text
Gobblet-Gobblers-AI-Agent/
│
├── Game-Project/          # Core game logic and board representation
│   ├── game.py            # Implements Gobblet Gobblers rules
│   ├── utils.py           # Helper functions
│   └── ...
│
├── Model-Project/         # AI models and agent logic
│   ├── agent.py           # Main AI decision-making agent
│   ├── training.py        # Model training or search algorithms
│   └── ...
│
├── Gobblet Gobblers.pdf   # Game rules and background material
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

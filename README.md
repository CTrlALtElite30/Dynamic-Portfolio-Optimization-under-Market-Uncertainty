# Deep Reinforcement Learning for Dynamic Portfolio Optimization  
A capstone project that looks at DQN and PPO agents. It focuses on managing stock portfolios in uncertain conditions through deep reinforcement learning.

# Overview  
Portfolio management is a basic challenge in finance. Traditional methods often have a hard time dealing with market volatility.  
This project uses **Deep Reinforcement Learning (DRL)** to dynamically improve portfolio allocation.  
We compare **DQN (Discrete actions)** and **PPO (Continuous actions)** using real-world stock price data.  

# Features
- Custom trading environments (TradingEnvDiscrete, TradingEnvContinuous)  
- Training with DQN and PPO agents (Stable-Baselines3)  
- Evaluation using financial metrics: Sharpe, Calmar, Sortino, CAGR  
- Visualization of equity curves and actions  
- Modular CLI pipeline through main.py  

# Project Structure
project/
│── data/                 # Raw stock data
│── envs/                 # Custom trading environments
│── utils/                # Utility functions (metrics, preprocessing)
│── outputs/
│   ├── datasets/         # Processed train/val/test datasets
│   ├── models/           # Saved DQN & PPO models
│   ├── evaluations/      # CSV + JSON evaluation results
│   └── plots/            # Visualization (equity curves etc.)
│── main.py               # Command-line interface
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation

# Installation
bash : 
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt # for all library installation 

# Usage

#Train both DQN and PPO
bash
python main.py --train both    # for separete use --train dqn
python main.py --train both --use-best
python main.py --eval all
python main.py --viz all (genarates every visuals)
python main.py --viz summary   (better - outputs\visuals) Rolling returns, Drawdowns, Cumulative returns (DQN vs PPO vs Baseline)
python main.py --all   # Run full pipeline


# Results (with plots + metrics)  
Add sample plots + metrics tables.
# Results

# Example Equity Curves
 

 


# Key Metrics (Test Set)
| Metric         | DQN       | PPO       |
|----------------|-----------|-----------|
| Total Return % | 15.2%     | 22.8%     |
| CAGR %         | 8.1%      | 11.4%     |
| Sharpe Ratio   | 0.78      | 1.05      |
| Max Drawdown % | -12.3%    | -9.5%     |


# Future Improvements
- Extend to multi-asset portfolios  
- Add risk-aware reward functions (e.g., CVaR)  
- Integrate live trading APIs (e.g., Alpaca, Interactive Brokers)  

# References
1. Singh V.; Chen S.-S.; Singhania M.; Nanavati B.; kar A.K.; Gupta A., How are reinforcement learning and deep learning algorithms used for big data based decision making in financial industries–A review and research agenda, 2022, International Journal of Information Management Data Insights, 2, 2, 10.1016/j.jjimei.2022.100094
2. Yang H.; Liu X.-Y.; Zhong S.; Walid A., Deep reinforcement learning for automated stock trading: An ensemble strategy, 2020, ICAIF 2020 - 1st ACM International Conference on AI in Finance, 10.1145/3383455.3422540
3. Soleymani F.; Paquet E., Financial portfolio optimization with online deep reinforcement learning and restricted stacked autoencoder—DeepBreath, 2020, Expert Systems with Applications, 156, 10.1016/j.eswa.2020.113456
4. Cappart Q.; Moisan T.; Rousseau L.-M.; Prémont-Schwarz I.; Cire A.A., Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization, 2021, 35th AAAI Conference on Artificial Intelligence, AAAI 2021, 5A, 3677.0, 3687.0, 10.1609/aaai.v35i5.16484
5. Hambly B.; Xu R.; Yang H., Recent advances in reinforcement learning in finance, 2023, Mathematical Finance, 33, 3, 437.0, 503.0, 10.1111/mafi.12382
6. Aboussalah A.M.; Lee C.-G., Continuous control with Stacked Deep Dynamic Recurrent Reinforcement Learning for portfolio optimization, 2020, Expert Systems with Applications, 140, 10.1016/j.eswa.2019.112891
7. Gutierrez-Franco E.; Mejia-Argueta C.; Rabelo L., Data-driven methodology to support long-lasting logistics and decision making for urban last-mile operations, 2021, Sustainability (Switzerland), 13, 11, 10.3390/su13116230
8. Polamuri S.R.; Srinivas D.K.; Krishna Mohan D.A., Multi-Model Generative Adversarial Network Hybrid Prediction Algorithm (MMGAN-HPA) for stock market prices prediction, 2022, Journal of King Saud University - Computer and Information Sciences, 34, 9, 7433.0, 7444.0, 10.1016/j.jksuci.2021.07.001
9. Lucarelli G.; Borrotti M., A deep Q-learning portfolio management framework for the cryptocurrency market, 2020, Neural Computing and Applications, 32, 23, 17229.0, 17244.0, 10.1007/s00521-020-05359-8
10. Schnaubelt M., Deep reinforcement learning for the optimal placement of cryptocurrency limit orders, 2022, European Journal of Operational Research, 296, 3, 993.0, 1006.0, 10.1016/j.ejor.2021.04.050


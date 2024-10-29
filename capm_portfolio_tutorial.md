## **In-Depth Tutorial: CAPM and Portfolio Optimization with Visualizations in Python**

This detailed tutorial explains every line of code to help you understand how to estimate the **required return of stocks using CAPM**, optimize a **portfolio** by maximizing the **Sharpe ratio**, and **visualize results** effectively. 

We’ll use **Python** to achieve the following:
1. Download market, stock, and risk-free data.
2. Calculate CAPM-required returns.
3. Estimate stock betas.
4. Optimize portfolio weights.
5. Visualize portfolio metrics and the **efficient frontier**.

---

## **Step-by-Step Code with Line-by-Line Explanation**

---

### **Step 1: Importing Required Libraries**

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from statsmodels.regression.rolling import RollingOLS
from statsmodels.api import add_constant
```

- **`pandas`**: Used for working with tabular data.
- **`numpy`**: For numerical operations (like matrix multiplication).
- **`yfinance`**: To fetch stock, market, and risk-free data from Yahoo Finance.
- **`matplotlib`** and **`seaborn`**: For data visualizations.
- **`scipy.optimize.minimize`**: For optimizing portfolio weights.
- **`statsmodels`**: To perform **rolling OLS regression** for beta estimation.

---

### **Step 2: Fetching Market and Risk-Free Data**

```python
market_data = yf.download('^GSPC', period='61mo', interval='1mo')
rf_data = yf.download('^IRX', period='61mo', interval='1mo')  # 3-month T-bill rate
```

- **Market Data (`^GSPC`)**: Downloading the **S&P 500 index** as the market benchmark.
- **Risk-Free Data (`^IRX`)**: Using the **3-month U.S. Treasury bill** as the risk-free rate.

**Output**: 
- `market_data`: Contains adjusted close prices for S&P 500.
- `rf_data`: Contains the 3-month T-bill rate.

---

### **Step 3: Calculating Monthly Returns and Excess Market Returns**

```python
market_data['Market_Return'] = market_data['Adj Close'].pct_change()
rf_data['Risk_Free'] = rf_data['Adj Close'] / 100

data = market_data[['Market_Return']].join(rf_data[['Risk_Free']], how='inner').dropna()
data['Excess_Market_Return'] = data['Market_Return'] - data['Risk_Free']
```

- **Market Return**: Calculated using the **percentage change** in adjusted closing prices.
- **Risk-Free Rate**: Converted from percentage to decimal.
- **Excess Market Return**: Calculated as `Market_Return - Risk_Free`.

---

### **Step 4: Fetching Stock Prices**

```python
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 
          'JPM', 'V', 'MA', 'DIS', 'PG', 'KO', 'PEP', 'CSCO', 'INTC', 'BA', 
          'XOM', 'WMT']

stock_data = yf.download(stocks, period='61mo', interval='1mo')['Adj Close']
```

- **Stock Prices**: Downloading **adjusted close prices** for the selected 20 stocks.

---

### **Step 5: Calculating Monthly Stock Returns**

```python
stock_returns = stock_data.pct_change().dropna()
aligned_returns = stock_returns.loc[data.index]
```

- **Monthly Returns**: Calculated for each stock using percentage change.
- **Aligning Data**: Ensuring stock and market data are aligned.

---

### **Step 6: Estimating Stock Betas Using Rolling OLS Regression**

```python
betas = {}
for stock in stocks:
    X = add_constant(data['Excess_Market_Return'])
    y = aligned_returns[stock] - data['Risk_Free']
    model = RollingOLS(y, X, window=60)
    results = model.fit()
    betas[stock] = results.params['Excess_Market_Return'].iloc[-1]
```

- **Rolling OLS**: For each stock, regress the **excess stock return** on the **excess market return** using a 60-month rolling window.
- **Beta Calculation**: The **beta** is the coefficient of `Excess_Market_Return`.

---

### **Step 7: Calculating Required Returns Using CAPM**

```python
next_year_rf = data['Risk_Free'].iloc[-1]
next_year_market_return = data['Market_Return'].mean()

required_returns = {stock: next_year_rf + betas[stock] * (next_year_market_return - next_year_rf) 
                    for stock in stocks}

plt.figure(figsize=(12, 6))
sns.barplot(x=list(required_returns.keys()), y=list(required_returns.values()), palette='Blues')
plt.xticks(rotation=45)
plt.title('Required Returns for Each Stock (Next Year)')
plt.ylabel('Required Return')
plt.xlabel('Stock')
plt.show()
```

- **CAPM Formula**:  
  \[ R_i = R_f + \beta_i \cdot (E(R_m) - R_f) \]  
- **Visualization**: A bar plot of the **required returns** for each stock.

---

### **Step 8: Portfolio Optimization (Maximizing Sharpe Ratio)**

```python
expected_returns = np.array(list(required_returns.values()))
cov_matrix = aligned_returns.cov().values

def portfolio_performance(weights):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights):
    portfolio_return, portfolio_volatility = portfolio_performance(weights)
    return -(portfolio_return - next_year_rf) / portfolio_volatility

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(len(stocks)))
initial_weights = np.array([1 / len(stocks)] * len(stocks))

result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
```

- **Performance Function**: Calculates portfolio return and volatility.
- **Negative Sharpe Ratio**: For **maximization**, we minimize the **negative** Sharpe ratio.
- **Constraints**: All weights sum to 1; no short selling (weights ≥ 0).
- **Optimization**: Using **SLSQP** to find optimal weights.

---

### **Step 9: Displaying Optimal Portfolio Weights**

```python
optimal_weights = result.x
opt_return, opt_volatility = portfolio_performance(optimal_weights)
opt_sharpe_ratio = (opt_return - next_year_rf) / opt_volatility

plt.figure(figsize=(12, 6))
sns.barplot(x=stocks, y=optimal_weights, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Optimal Portfolio Weights')
plt.ylabel('Weight')
plt.xlabel('Stock')
plt.show()

print(f"\nExpected Portfolio Return: {opt_return:.2%}")
print(f"Expected Portfolio Volatility: {opt_volatility:.2%}")
print(f"Expected Portfolio Sharpe Ratio: {opt_sharpe_ratio:.2f}")
```

- **Bar Plot**: Visualizes the **optimal portfolio weights**.
- **Portfolio Metrics**: Displays the **expected return, volatility, and Sharpe ratio**.

---

### **Step 10: Plotting the Efficient Frontier**

```python
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    portfolio_return, portfolio_volatility = portfolio_performance(weights)
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = (portfolio_return - next_year_rf) / portfolio_volatility

plt.figure(figsize=(12, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(opt_volatility, opt_return, color='red', marker='*', s=200, label='Optimal Portfolio')
plt.title('Efficient Frontier with Optimal Portfolio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend(loc='best')
plt.show()
```

- **Efficient Frontier**: Visualizes the **risk-return trade-off** for 5,000 random portfolios.
- **Optimal Portfolio**: Highlighted with a **red star**.

---

### **Conclusion**

This tutorial covers:
- CAPM calculations.
- Portfolio optimization using the Sharpe ratio.
- Visualizing optimal weights and the efficient frontier.

You now have a complete, hands-on approach to **CAPM and portfolio optimization** using Python.
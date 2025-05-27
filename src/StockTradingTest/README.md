# Stock Trading Test System

A production-ready console application for testing and evaluating different AI models for stock price prediction and automated trading.

## Features

- Test multiple IFullModel implementations from AiDotNet in a tournament-style competition
- Production-ready design for real stock market trading
- Comprehensive performance metrics (profit/loss, Sharpe ratio, win rate, etc.)
- Configurable trading strategy with stop-loss and take-profit controls
- Detailed performance reports with charts
- Logging system for trade execution and simulation tracking

## Configuration

The application is configured through `appsettings.json` with the following sections:

### Trading Simulation Settings

```json
"TradingSimulation": {
  "InitialBalance": 10000.00,
  "MaxPositions": 5,
  "SimulationDays": 30,
  "CommissionPerTrade": 0.001,
  "MaxPositionSizePercent": 0.2,
  "StopLossPercent": 0.05,
  "TakeProfitPercent": 0.1,
  "RiskPerTradePercent": 0.02
}
```

### Model Competition Settings

```json
"ModelCompetition": {
  "NumberOfRounds": 3,
  "EliminationPercentage": 0.5,
  "ValidationDataSplit": 0.2,
  "TestDataSplit": 0.2,
  "LookbackPeriod": 60,
  "PredictionHorizon": 5
}
```

### Data Source Settings

```json
"DataSource": {
  "StockDataPath": "Data/StockData",
  "DefaultSymbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "StartDate": "2020-01-01",
  "EndDate": "2023-12-31",
  "FeaturesToInclude": ["Open", "High", "Low", "Close", "Volume"]
}
```

### Logging Settings

```json
"Logging": {
  "LogLevel": "Information",
  "LogToFile": true,
  "LogFilePath": "Logs/trading_simulation.log"
}
```

## Running the Application

1. Place historical stock data CSV files in the `Data/StockData` directory with filenames matching the stock symbols.
2. Adjust configuration settings in `appsettings.json` as needed.
3. Run the application:

```
dotnet run
```

## Data Format

The application expects CSV files with the following columns:
- Date
- Open
- High
- Low
- Close
- Adj Close (optional)
- Volume

Example data sources:
- Yahoo Finance historical data
- Alpha Vantage API
- IEX Cloud API

## Model Types Supported

The system tests the following models from AiDotNet:
- Feed-forward Neural Networks
- LSTM Networks
- Transformer Models
- Random Forests
- Gradient Boosting
- Support Vector Regression
- Gaussian Processes
- ARIMA Models

## Example Output

The application will generate:
1. Console output with round-by-round results and final rankings
2. A detailed HTML report with performance charts
3. Log files documenting the simulation process

## Extending the System

### Adding New Models

To add a new model type:
1. Create a new method in the `ModelTrainer` class
2. Configure the model using `PredictionModelBuilder`
3. Add it to the `CreateCompetitors` method

### Custom Trading Strategies

The trading logic can be customized in the `MakeTradingDecisions` method of the `TradingSimulationService` class.
---
title: "Finance"
description: "All 169 public types in the AiDotNet.finance namespace, organized by kind."
section: "API Reference"
---

**169** public types in this namespace, organized by kind.

## Models & Types (130)

| Type | Summary |
|:-----|:--------|
| [`AlphaFactorModel<T>`](/docs/reference/wiki/finance/alphafactormodel/) | Neural network model for learning alpha factors from market data. |
| [`AttentionAllocation<T>`](/docs/reference/wiki/finance/attentionallocation/) | Portfolio optimizer using Multi-Head Attention to capture asset relationships. |
| [`Autoformer<T>`](/docs/reference/wiki/finance/autoformer/) | Autoformer (Decomposition Transformers with Auto-Correlation) neural network for time series forecasting. |
| [`BlackLittermanNeural<T>`](/docs/reference/wiki/finance/blacklittermanneural/) | Neural network adaptation of the Black-Litterman model. |
| [`BlackScholes<T>`](/docs/reference/wiki/finance/blackscholes/) | Closed-form Black-Scholes-Merton European option pricing, the full first-order Greeks (delta, gamma, vega, theta, rho) and implied-volatility solve. |
| [`BloombergGPT<T>`](/docs/reference/wiki/finance/bloomberggpt/) | BloombergGPT neural network model for comprehensive financial language processing. |
| [`BrokerOptionsProfile`](/docs/reference/wiki/finance/brokeroptionsprofile/) | A broker's mapping from the stable `OptionStrategyClass` archetypes to its own approval `OptionsApprovalLevel` numbers, plus the level this account is authorized for. |
| [`CCDM<T>`](/docs/reference/wiki/finance/ccdm/) | CCDM — Conditional Continuous Diffusion Model for Time Series. |
| [`CSDI<T>`](/docs/reference/wiki/finance/csdi/) | CSDI — Conditional Score-based Diffusion Model for Probabilistic Time Series Imputation. |
| [`CSDI<T>`](/docs/reference/wiki/finance/csdi-2/) | CSDI (Conditional Score-based Diffusion model for Imputation) for probabilistic time series imputation. |
| [`ChronosBolt<T>`](/docs/reference/wiki/finance/chronosbolt/) | Chronos-Bolt — Fast Non-Autoregressive Time Series Forecasting from the Chronos Family. |
| [`Chronos<T>`](/docs/reference/wiki/finance/chronos/) | Chronos foundation model for time series forecasting using tokenization. |
| [`Crossformer<T>`](/docs/reference/wiki/finance/crossformer/) | Crossformer (Cross-Dimension Transformer) neural network for multivariate time series forecasting. |
| [`DCRNN<T>`](/docs/reference/wiki/finance/dcrnn/) | DCRNN (Diffusion Convolutional Recurrent Neural Network) for spatial-temporal forecasting. |
| [`DeepAR<T>`](/docs/reference/wiki/finance/deepar/) | DeepAR probabilistic autoregressive forecasting model using LSTM networks. |
| [`DeepFactor<T>`](/docs/reference/wiki/finance/deepfactor/) | DeepFactor (Deep Factor Model) for multivariate time series forecasting. |
| [`DeepPortfolioManager<T>`](/docs/reference/wiki/finance/deepportfoliomanager/) | Deep Portfolio Manager for end-to-end portfolio weight optimization. |
| [`DeepState<T>`](/docs/reference/wiki/finance/deepstate/) | DeepState (Deep State Space Model) for probabilistic time series forecasting. |
| [`DiffusionTS<T>`](/docs/reference/wiki/finance/diffusionts/) | DiffusionTS (Interpretable Diffusion for Time Series) for probabilistic forecasting with seasonal-trend decomposition. |
| [`EGarchModel<T>`](/docs/reference/wiki/finance/egarchmodel/) | EGARCH(1,1) — Nelson (1991, "Conditional Heteroskedasticity in Asset Returns: A New Approach", Econometrica 59). |
| [`ETSformer<T>`](/docs/reference/wiki/finance/etsformer/) | ETSformer: Exponential Smoothing Transformer for time series forecasting. |
| [`FEDformer<T>`](/docs/reference/wiki/finance/fedformer/) | FEDformer (Frequency Enhanced Decomposed Transformer) for long-term time series forecasting. |
| [`FactorTransformer<T>`](/docs/reference/wiki/finance/factortransformer/) | Transformer-based model for learning financial factors with attention mechanisms. |
| [`FactorVAE<T>`](/docs/reference/wiki/finance/factorvae/) | Variational autoencoder for learning disentangled financial factors. |
| [`FinBERTTone<T>`](/docs/reference/wiki/finance/finberttone/) | FinBERT-tone neural network model specialized for financial sentiment analysis. |
| [`FinBERT<T>`](/docs/reference/wiki/finance/finbert/) | FinBERT (Financial BERT) model for financial sentiment analysis. |
| [`FinGPT<T>`](/docs/reference/wiki/finance/fingpt/) | FinGPT neural network model for domain-specific financial language generation and analysis. |
| [`FinMA<T>`](/docs/reference/wiki/finance/finma/) | FinMA (Financial Multi-Agent) neural network model for collaborative financial task solving. |
| [`FinRLAgent<T>`](/docs/reference/wiki/finance/finrlagent/) | Unified FinRL-style agent that can switch between multiple RL algorithms. |
| [`FinancialA2CAgent<T>`](/docs/reference/wiki/finance/financiala2cagent/) | Financial Advantage Actor-Critic (A2C) agent for fast trading policy learning. |
| [`FinancialAutoML<T>`](/docs/reference/wiki/finance/financialautoml/) | AutoML implementation for finance models (forecasting and risk). |
| [`FinancialBERT<T>`](/docs/reference/wiki/finance/financialbert/) | FinancialBERT neural network model for domain-specific financial language processing. |
| [`FinancialDQNAgent<T>`](/docs/reference/wiki/finance/financialdqnagent/) | Financial Deep Q-Network (DQN) agent for discrete action trading. |
| [`FinancialDataLoader<T>`](/docs/reference/wiki/finance/financialdataloader/) | Data loader for financial time series forecasting. |
| [`FinancialPPOAgent<T>`](/docs/reference/wiki/finance/financialppoagent/) | Financial Proximal Policy Optimization (PPO) agent for robust trading. |
| [`FinancialPreprocessor<T>`](/docs/reference/wiki/finance/financialpreprocessor/) | Preprocesses financial time series data into model-ready tensors. |
| [`FinancialSACAgent<T>`](/docs/reference/wiki/finance/financialsacagent/) | Financial Soft Actor-Critic (SAC) agent for high-performance continuous trading. |
| [`FinancialSearchSpace`](/docs/reference/wiki/finance/financialsearchspace/) | Provides default AutoML search spaces for finance models. |
| [`FlowState<T>`](/docs/reference/wiki/finance/flowstate/) | FlowState — IBM's SSM-based Time Series Foundation Model (9.1M parameters). |
| [`Fold`](/docs/reference/wiki/finance/fold/) | One purged/embargoed walk-forward fold: the surviving train indices and the test indices. |
| [`GIFTEvalBenchmark<T>`](/docs/reference/wiki/finance/giftevalbenchmark/) | GIFT-Eval benchmark implementation for standardized evaluation of time series foundation models. |
| [`GPT4TS<T>`](/docs/reference/wiki/finance/gpt4ts/) | GPT4TS — One Fits All: Power General Time Series Analysis by Pretrained LM. |
| [`Garch11Model<T>`](/docs/reference/wiki/finance/garch11model/) | GARCH(1,1) — Bollerslev (1986, "Generalized Autoregressive Conditional Heteroskedasticity", Journal of Econometrics 31). |
| [`GjrGarchModel<T>`](/docs/reference/wiki/finance/gjrgarchmodel/) | GJR-GARCH(1,1,1) — Glosten, Jagannathan & Runkle (1993, "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks", Journal of Finance 48). |
| [`GraphWaveNet<T>`](/docs/reference/wiki/finance/graphwavenet/) | GraphWaveNet (Graph WaveNet) for deep spatial-temporal graph modeling. |
| [`HarRvModel<T>`](/docs/reference/wiki/finance/harrvmodel/) | HAR-RV — the Heterogeneous AutoRegressive model of Realized Volatility (Corsi, 2009, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics 7(2)). |
| [`HierarchicalRiskParity<T>`](/docs/reference/wiki/finance/hierarchicalriskparity/) | Neural Hierarchical Risk Parity (HRP) for portfolio optimization. |
| [`Hippo<T>`](/docs/reference/wiki/finance/hippo/) | HiPPO (High-order Polynomial Projection Operators) for time series forecasting. |
| [`ITransformer<T>`](/docs/reference/wiki/finance/itransformer/) | iTransformer (Inverted Transformer) neural network for time series forecasting. |
| [`Informer<T>`](/docs/reference/wiki/finance/informer/) | Informer (Efficient Transformer for Long Sequence Forecasting) neural network. |
| [`Interval<T>`](/docs/reference/wiki/finance/interval/) | The percentile confidence interval plus the point estimate computed on the full sample. |
| [`InvestLM<T>`](/docs/reference/wiki/finance/investlm/) | InvestLM neural network model specialized for investment professionals and research. |
| [`Kairos<T>`](/docs/reference/wiki/finance/kairos/) | Kairos — Adaptive and Generalizable Time Series Foundation Model with Mixture-of-Size Encoder. |
| [`KellyCriterion<T>`](/docs/reference/wiki/finance/kellycriterion/) | Kelly-criterion position sizing — the bet fraction that maximizes long-run log-growth of capital. |
| [`Kronos<T>`](/docs/reference/wiki/finance/kronos/) | Kronos — Foundation Model for the Language of Financial Markets. |
| [`LLMTime<T>`](/docs/reference/wiki/finance/llmtime/) | LLM-Time — Zero-Shot Time Series Forecasting via LLM Tokenization. |
| [`LSTNet<T>`](/docs/reference/wiki/finance/lstnet/) | LSTNet (Long Short-Term Time-series Network) model for multivariate time series forecasting. |
| [`LagLlama<T>`](/docs/reference/wiki/finance/lagllama/) | Lag-Llama foundation model for probabilistic time series forecasting. |
| [`MGTSD<T>`](/docs/reference/wiki/finance/mgtsd/) | MG-TSD — Multi-Granularity Time Series Diffusion Model with Guided Learning. |
| [`MOIRAI<T>`](/docs/reference/wiki/finance/moirai/) | MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) implementation. |
| [`MOMENT<T>`](/docs/reference/wiki/finance/moment/) | MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model. |
| [`MQCNN<T>`](/docs/reference/wiki/finance/mqcnn/) | MQCNN (Multi-Quantile Convolutional Neural Network) for probabilistic time series forecasting. |
| [`MTGNN<T>`](/docs/reference/wiki/finance/mtgnn/) | MTGNN (Multivariate Time-series Graph Neural Network) for automatic graph learning and forecasting. |
| [`Mamba2<T>`](/docs/reference/wiki/finance/mamba2/) | Mamba-2 (State Space Duality) implementation for time series forecasting. |
| [`Mamba<T>`](/docs/reference/wiki/finance/mamba/) | Mamba (Selective State Space Model) implementation for time series forecasting. |
| [`MarketDataPoint<T>`](/docs/reference/wiki/finance/marketdatapoint/) | Represents a single market data point (OHLCV). |
| [`MarketDataProvider<T>`](/docs/reference/wiki/finance/marketdataprovider/) | Stores and serves market data for finance workflows. |
| [`MarketMakingAgent<T>`](/docs/reference/wiki/finance/marketmakingagent/) | Specialized market making agent using reinforcement learning for optimal quoting. |
| [`MarketMakingEnvironment<T>`](/docs/reference/wiki/finance/marketmakingenvironment/) | Market making environment that simulates bid/ask quoting and inventory risk. |
| [`MarkowitzOptimizer<T>`](/docs/reference/wiki/finance/markowitzoptimizer/) | Closed-form Markowitz mean-variance portfolio optimization: the global minimum-variance portfolio, the tangency (maximum-Sharpe) portfolio, and the efficient-frontier target-return portfolio. |
| [`MultiTaskEvaluationHarness<T>`](/docs/reference/wiki/finance/multitaskevaluationharness/) | Multi-task evaluation harness for time series foundation models, supporting standardized evaluation across forecasting, anomaly detection, classification, imputation, and embedding tasks. |
| [`NBEATSFinance<T>`](/docs/reference/wiki/finance/nbeatsfinance/) | N-BEATS (Neural Basis Expansion Analysis for Time Series) model for financial forecasting. |
| [`NHiTSFinance<T>`](/docs/reference/wiki/finance/nhitsfinance/) | N-HiTS (Neural Hierarchical Interpolation for Time Series) model for financial forecasting. |
| [`NeuralCVaR<T>`](/docs/reference/wiki/finance/neuralcvar/) | A neural network model for estimating Conditional Value at Risk (CVaR), also known as Expected Shortfall. |
| [`NeuralGARCH<T>`](/docs/reference/wiki/finance/neuralgarch/) | Neural GARCH model for forecasting asset volatility. |
| [`NeuralStressTest<T>`](/docs/reference/wiki/finance/neuralstresstest/) | Neural network model for generating and evaluating stress test scenarios. |
| [`NeuralVaR<T>`](/docs/reference/wiki/finance/neuralvar/) | Neural Value-at-Risk (VaR) model for non-linear market risk assessment. |
| [`NonStationaryTransformer<T>`](/docs/reference/wiki/finance/nonstationarytransformer/) | Non-stationary Transformer neural network for time series forecasting with changing statistics. |
| [`OptionLeg`](/docs/reference/wiki/finance/optionleg/) | One option leg of a strategy. |
| [`OptionStrategy`](/docs/reference/wiki/finance/optionstrategy/) | A named option strategy: a set of `OptionLeg`s (plus an optional `StockLeg`), with the broker approval level it requires, whether its risk is defined, and its expiry payoff profile (max loss, max gain, breakevens) computed by sampling the p… |
| [`PatchTST<T>`](/docs/reference/wiki/finance/patchtst/) | PatchTST (Patch Time Series Transformer) neural network for long-term time series forecasting. |
| [`PortfolioTradingEnvironment<T>`](/docs/reference/wiki/finance/portfoliotradingenvironment/) | Multi-asset portfolio trading environment with continuous weight actions. |
| [`QuantileForecastResult<T>`](/docs/reference/wiki/finance/quantileforecastresult/) | Represents the result of a probabilistic/quantile forecast from a time series foundation model. |
| [`RWKVForecaster<T>`](/docs/reference/wiki/finance/rwkvforecaster/) | RWKV (Receptance Weighted Key Value) implementation for time series forecasting. |
| [`RealizedVolatilityTransformer<T>`](/docs/reference/wiki/finance/realizedvolatilitytransformer/) | Realized Volatility Transformer for attention-based volatility forecasting. |
| [`RelationalGCN<T>`](/docs/reference/wiki/finance/relationalgcn/) | RelationalGCN (Relational Graph Convolutional Network) for multi-relational graph learning. |
| [`Result`](/docs/reference/wiki/finance/result/) | Result of the BH procedure: per-hypothesis rejection flags and adjusted q-values. |
| [`RiskRatios<T>`](/docs/reference/wiki/finance/riskratios/) | Risk-adjusted performance ratios from a periodic return series: Sharpe, Sortino, and Calmar. |
| [`S4<T>`](/docs/reference/wiki/finance/s4/) | S4 (Structured State Space Sequence Model) for time series forecasting. |
| [`SAINT<T>`](/docs/reference/wiki/finance/saint/) | SAINT (Self-Attention and Intersample Attention Transformer) for tabular data. |
| [`SECBERT<T>`](/docs/reference/wiki/finance/secbert/) | SEC-BERT neural network model for domain-specific financial language processing. |
| [`STGNN<T>`](/docs/reference/wiki/finance/stgnn/) | STGNN (Spatio-Temporal Graph Neural Network) for forecasting on graph-structured time series data. |
| [`ScoreGrad<T>`](/docs/reference/wiki/finance/scoregrad/) | ScoreGrad (Score-based Gradient Model) for probabilistic time series forecasting using score matching. |
| [`SentimentResult<T>`](/docs/reference/wiki/finance/sentimentresult/) | Result of sentiment analysis on a single text. |
| [`SimMTM<T>`](/docs/reference/wiki/finance/simmtm/) | SimMTM — Simple Pre-Training Framework for Masked Time-Series Modeling. |
| [`StockLeg`](/docs/reference/wiki/finance/stockleg/) | An optional stock leg (covered strategies pair stock with options). |
| [`StockTradingEnvironment<T>`](/docs/reference/wiki/finance/stocktradingenvironment/) | Simple single-asset trading environment with buy/hold/sell actions. |
| [`Sundial<T>`](/docs/reference/wiki/finance/sundial/) | Sundial — A Family of Highly Capable Time Series Foundation Models. |
| [`TCN<T>`](/docs/reference/wiki/finance/tcn/) | TCN (Temporal Convolutional Network) model for time series forecasting. |
| [`TEST<T>`](/docs/reference/wiki/finance/test/) | TEST — Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series. |
| [`TFC<T>`](/docs/reference/wiki/finance/tfc/) | TF-C — Time-Frequency Consistency for Self-Supervised Time Series. |
| [`TFT<T>`](/docs/reference/wiki/finance/tft/) | Temporal Fusion Transformer (TFT) neural network for multi-horizon time series forecasting. |
| [`TOTEM<T>`](/docs/reference/wiki/finance/totem/) | TOTEM — TOkenized Time Series EMbeddings via VQ-VAE. |
| [`TOTO<T>`](/docs/reference/wiki/finance/toto/) | TOTO — Datadog's Time Series Foundation Model for Observability. |
| [`TS2Vec<T>`](/docs/reference/wiki/finance/ts2vec/) | TS2Vec — Contrastive Learning of Universal Time Series Representations. |
| [`TSDiff<T>`](/docs/reference/wiki/finance/tsdiff/) | TSDiff — Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting. |
| [`TSDiff<T>`](/docs/reference/wiki/finance/tsdiff-2/) | TSDiff (Time Series Diffusion) for probabilistic time series forecasting with self-guided diffusion. |
| [`TSMixer<T>`](/docs/reference/wiki/finance/tsmixer/) | TSMixer: An all-MLP architecture for time series forecasting. |
| [`TabNet<T>`](/docs/reference/wiki/finance/tabnet/) | TabNet model for tabular data learning, combining tree-based interpretability with deep learning performance. |
| [`TabTransformer<T>`](/docs/reference/wiki/finance/tabtransformer/) | TabTransformer model for tabular data, using transformers for categorical features. |
| [`TemporalGCN<T>`](/docs/reference/wiki/finance/temporalgcn/) | TemporalGCN (Temporal Graph Convolutional Network) for time series forecasting on graph-structured data. |
| [`TimeBridge<T>`](/docs/reference/wiki/finance/timebridge/) | TimeBridge — Non-Stationarity Matters for Time Series Foundation Models. |
| [`TimeDiff<T>`](/docs/reference/wiki/finance/timediff/) | TimeDiff — Non-autoregressive Conditional Diffusion Models for Time Series Prediction. |
| [`TimeGPT<T>`](/docs/reference/wiki/finance/timegpt/) | TimeGPT-style time series foundation model implementation. |
| [`TimeGrad<T>`](/docs/reference/wiki/finance/timegrad/) | TimeGrad — Autoregressive Denoising Diffusion Model for Time Series Forecasting. |
| [`TimeGrad<T>`](/docs/reference/wiki/finance/timegrad-2/) | TimeGrad (Autoregressive Denoising Diffusion Model) for probabilistic time series forecasting. |
| [`TimeLLM<T>`](/docs/reference/wiki/finance/timellm/) | Time-LLM (Large Language Model Reprogramming for Time Series) implementation. |
| [`TimeMAE<T>`](/docs/reference/wiki/finance/timemae/) | TimeMAE — Masked Autoencoder for Time Series with Decoupled Masked Autoencoders. |
| [`TimeMachine<T>`](/docs/reference/wiki/finance/timemachine/) | TimeMachine (Time Series State Space Model) for multi-scale time series forecasting. |
| [`TimeMoE<T>`](/docs/reference/wiki/finance/timemoe/) | Time-MoE — Billion-Scale Time Series Foundation Models with Mixture of Experts. |
| [`TimeSeriesTokenizer<T>`](/docs/reference/wiki/finance/timeseriestokenizer/) | Standard tokenization pipeline for time series foundation models, supporting patching, quantization, and instance normalization strategies. |
| [`Timer<T>`](/docs/reference/wiki/finance/timer/) | Timer (Generative Pre-Training for Time Series) implementation. |
| [`TimesFM<T>`](/docs/reference/wiki/finance/timesfm/) | TimesFM (Time Series Foundation Model) for zero-shot time series forecasting. |
| [`TimesNet<T>`](/docs/reference/wiki/finance/timesnet/) | TimesNet (Temporal 2D-Variation Modeling) neural network for time series analysis. |
| [`TinyTimeMixers<T>`](/docs/reference/wiki/finance/tinytimemixers/) | Tiny Time Mixers (TTM) foundation model for compact, high-performance time series forecasting. |
| [`UniTS<T>`](/docs/reference/wiki/finance/units/) | UniTS (Unified Time Series Model) implementation for multi-task time series processing. |
| [`VisionTS<T>`](/docs/reference/wiki/finance/visionts/) | VisionTS — Visual Masked Autoencoders as Zero-Shot Time Series Forecasters. |
| [`VolEdge`](/docs/reference/wiki/finance/voledge/) | The volatility edge: our FORECAST realized vol vs the option market's IMPLIED vol, and the resulting stance. |
| [`WaveNet<T>`](/docs/reference/wiki/finance/wavenet/) | WaveNet model adapted for time series forecasting. |
| [`YingLong<T>`](/docs/reference/wiki/finance/yinglong/) | YingLong — Alibaba's Enterprise Time Series Foundation Model. |

## Base Classes (9)

| Type | Summary |
|:-----|:--------|
| [`ClassicalVolatilityModelBase<T>`](/docs/reference/wiki/finance/classicalvolatilitymodelbase/) | Base for CLASSICAL (econometric) conditional-volatility models — GARCH and its variants — estimated by **maximum likelihood**, exactly as in their original papers, rather than by neural-network training. |
| [`FinancialModelBase<T>`](/docs/reference/wiki/finance/financialmodelbase/) | Base class for all financial AI models, providing dual ONNX/native mode support. |
| [`FinancialNLPModelBase<T>`](/docs/reference/wiki/finance/financialnlpmodelbase/) | Base class for all financial NLP models, implementing the dual-mode pattern. |
| [`ForecastingModelBase<T>`](/docs/reference/wiki/finance/forecastingmodelbase/) | Base class for financial forecasting models, adding forecasting-specific behavior on top of the core financial model infrastructure. |
| [`PortfolioOptimizerBase<T>`](/docs/reference/wiki/finance/portfoliooptimizerbase/) | Base class for portfolio optimization models. |
| [`RiskModelBase<T>`](/docs/reference/wiki/finance/riskmodelbase/) | Base class for risk management models, providing common infrastructure for risk assessment. |
| [`TimeSeriesFoundationModelBase<T>`](/docs/reference/wiki/finance/timeseriesfoundationmodelbase/) | Abstract base class for time series foundation models that support multiple downstream tasks. |
| [`TradingAgentBase<T>`](/docs/reference/wiki/finance/tradingagentbase/) | Base class for financial trading agents using reinforcement learning. |
| [`TradingEnvironment<T>`](/docs/reference/wiki/finance/tradingenvironment/) | Base environment for financial trading simulations. |

## Interfaces (13)

| Type | Summary |
|:-----|:--------|
| [`IFactorModel<T>`](/docs/reference/wiki/finance/ifactormodel/) | Interface for financial factor models that learn latent factors from market data. |
| [`IFinancialModel<T>`](/docs/reference/wiki/finance/ifinancialmodel/) | Base interface for all financial AI models in AiDotNet.Finance. |
| [`IFinancialNLPModel<T>`](/docs/reference/wiki/finance/ifinancialnlpmodel/) |  |
| [`IForecastingModel<T>`](/docs/reference/wiki/finance/iforecastingmodel/) | Interface for time series forecasting models in the Finance module. |
| [`IMeanVarianceOptimizer<T>`](/docs/reference/wiki/finance/imeanvarianceoptimizer/) | A closed-form mean-variance portfolio optimizer: the analytic (training-free) weight solutions. |
| [`IOptionPricer<T>`](/docs/reference/wiki/finance/ioptionpricer/) | A European-option pricing engine: fair value, first-order Greeks, and implied volatility. |
| [`IPortfolioOptimizer<T>`](/docs/reference/wiki/finance/iportfoliooptimizer/) | Interface for portfolio optimization models that determine optimal asset allocations. |
| [`IPositionSizer<T>`](/docs/reference/wiki/finance/ipositionsizer/) | A bet/position-sizing rule: given an edge, returns the fraction of capital to allocate. |
| [`IRiskModel<T>`](/docs/reference/wiki/finance/iriskmodel/) | Interface for financial risk models that estimate potential losses. |
| [`IRiskRatioCalculator<T>`](/docs/reference/wiki/finance/iriskratiocalculator/) | Computes risk-adjusted performance ratios (Sharpe, Sortino, Calmar) from a periodic return series. |
| [`ITimeSeriesFoundationModel<T>`](/docs/reference/wiki/finance/itimeseriesfoundationmodel/) | Interface for multi-task time series foundation models that support forecasting, anomaly detection, classification, imputation, and embedding generation. |
| [`ITradingAgent<T>`](/docs/reference/wiki/finance/itradingagent/) | Interface for financial trading agents using reinforcement learning. |
| [`IVolatilityModel<T>`](/docs/reference/wiki/finance/ivolatilitymodel/) | Interface for volatility models that forecast price variability. |

## Enums (8)

| Type | Summary |
|:-----|:--------|
| [`FinRLAlgorithm`](/docs/reference/wiki/finance/finrlalgorithm/) | RL algorithms supported by FinRLAgent. |
| [`FinancialDomain`](/docs/reference/wiki/finance/financialdomain/) | Defines which finance sub-domain AutoML should focus on. |
| [`FinancialNLPTaskType`](/docs/reference/wiki/finance/financialnlptasktype/) | Defines the different types of NLP tasks for financial text processing. |
| [`OptionRight`](/docs/reference/wiki/finance/optionright/) | Call or put. |
| [`OptionStrategyClass`](/docs/reference/wiki/finance/optionstrategyclass/) | Broker-INDEPENDENT risk archetype of an option strategy — what the position actually IS, regardless of how any particular broker numbers its approval tiers. |
| [`OptionTradeAction`](/docs/reference/wiki/finance/optiontradeaction/) | Buy (long) or sell (short) a leg. |
| [`OptionsApprovalLevel`](/docs/reference/wiki/finance/optionsapprovallevel/) | Broker options-approval tiers. |
| [`VolStance`](/docs/reference/wiki/finance/volstance/) | Whether the vol edge says to sell, buy, or sit out volatility. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`ClassicalVolatilityModelOptions`](/docs/reference/wiki/finance/classicalvolatilitymodeloptions/) | Options for the classical (econometric) volatility models. |

## Helpers & Utilities (8)

| Type | Summary |
|:-----|:--------|
| [`BenjaminiHochbergFdr`](/docs/reference/wiki/finance/benjaminihochbergfdr/) | Benjamini-Hochberg false-discovery-rate (FDR) control for multiple hypothesis testing. |
| [`BootstrapConfidenceInterval<T>`](/docs/reference/wiki/finance/bootstrapconfidenceinterval/) | Bootstrap confidence intervals for a statistic of a return series. |
| [`DeflatedSharpeRatio<T>`](/docs/reference/wiki/finance/deflatedsharperatio/) | López de Prado's Deflated Sharpe Ratio (DSR): the probability that a strategy's *true* Sharpe ratio is positive, after deflating the observed Sharpe for (a) the number of strategy configurations tried (multiple-testing / selection bias) and… |
| [`FinancialDataLoaderFactory`](/docs/reference/wiki/finance/financialdataloaderfactory/) | Factory helpers for creating financial data loaders. |
| [`InformationCoefficient<T>`](/docs/reference/wiki/finance/informationcoefficient/) | Information Coefficient (IC): the correlation between predicted and realized forward returns, plus its statistical significance (t-statistic, two-sided p-value) and the IC information ratio (ICIR) over a time series of per-period ICs. |
| [`PurgedWalkForwardValidator`](/docs/reference/wiki/finance/purgedwalkforwardvalidator/) | López de Prado purged-and-embargoed walk-forward cross-validation for time-ordered financial data. |
| [`TradingEnvironmentFactory`](/docs/reference/wiki/finance/tradingenvironmentfactory/) | Factory helpers for creating trading environments from market data. |
| [`VolatilityOptionsSignal`](/docs/reference/wiki/finance/volatilityoptionssignal/) | Turns a realized-volatility FORECAST (the one signal that is actually predictable — see the platform's vol research) into an options stance and a concrete DEFINED-RISK structure, by comparing it to the option market's implied vol:  - foreca… |


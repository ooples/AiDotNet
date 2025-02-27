namespace AiDotNet.Enums;

public enum TimeSeriesModelType
{
    // Auto-Regressive Integrated Moving Average models
    ARIMA,
    SARIMA, // Seasonal ARIMA
    ARMA,   // Auto-Regressive Moving Average
    AR,     // Auto-Regressive
    MA,     // Moving Average

    // Exponential Smoothing models
    ExponentialSmoothing,
    SimpleExponentialSmoothing,
    DoubleExponentialSmoothing, // Also known as Holt's method
    TripleExponentialSmoothing, // Also known as Holt-Winters' method

    // State Space models
    StateSpace,
    TBATS, // Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components

    // Regression-based models
    DynamicRegressionWithARIMAErrors,
    ARIMAX, // ARIMA with exogenous variables

    // Advanced models
    GARCH,  // Generalized Autoregressive Conditional Heteroskedasticity
    VAR,    // Vector Autoregression
    VARMA,  // Vector Autoregression Moving-Average

    // Machine Learning based models
    ProphetModel, // Facebook's Prophet model
    NeuralNetworkARIMA, // Neural Network ARIMA hybrid

    // Bayesian models
    BayesianStructuralTimeSeriesModel,

    // Spectral Analysis models
    SpectralAnalysis,

    // Decomposition models
    STLDecomposition, // Seasonal and Trend decomposition using Loess

    // Other specialized models
    InterventionAnalysis,
    TransferFunctionModel,
    UnobservedComponentsModel,

    // Custom or user-defined model
    Custom
}
using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the DeepState (Deep State Space) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DeepState combines deep learning with classical state space models (SSM) for probabilistic
/// time series forecasting. A neural network learns to parameterize the state space model,
/// while the SSM structure provides interpretable components like trend and seasonality.
/// </para>
/// <para><b>For Beginners:</b> DeepState is like giving a classical statistics model a brain:
///
/// <b>What is a State Space Model?</b>
/// SSMs assume your data has hidden "states" that evolve over time:
/// - State transition: z_t = F * z_{t-1} + noise (how states evolve)
/// - Observation: y_t = H * z_t + noise (how states produce observations)
///
/// <b>Example - Trend + Seasonality:</b>
/// States might represent:
/// - Level (current baseline)
/// - Trend (direction of change)
/// - Seasonal patterns (weekly/yearly cycles)
/// The observed value is a combination of these hidden states.
///
/// <b>Why "Deep" State Space?</b>
/// Classical SSMs need you to specify the model structure (how many seasonal patterns, etc.).
/// DeepState uses a neural network to:
/// - Automatically learn appropriate state representations
/// - Adapt to complex, non-linear patterns
/// - Share patterns across multiple time series
///
/// <b>Benefits:</b>
/// - Interpretable decomposition (trend, seasonality, residual)
/// - Natural uncertainty quantification
/// - Handles multiple related time series well
/// - Works with irregular data and missing values
/// </para>
/// <para>
/// <b>Reference:</b> Rangapuram et al., "Deep State Space Models for Time Series Forecasting", 2018.
/// https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
/// </para>
/// </remarks>
public class DeepStateOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DeepStateOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default DeepState configuration suitable for
    /// common forecasting tasks with both trend and seasonal components.
    /// </para>
    /// </remarks>
    public DeepStateOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of existing options, useful when you want
    /// to try variations of a working configuration.
    /// </para>
    /// </remarks>
    public DeepStateOptions(DeepStateOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        StateDimension = other.StateDimension;
        HiddenDimension = other.HiddenDimension;
        NumRnnLayers = other.NumRnnLayers;
        SeasonalPeriods = other.SeasonalPeriods;
        UseTrend = other.UseTrend;
        UseSeasonality = other.UseSeasonality;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (input sequence length).
    /// </summary>
    /// <value>The lookback window, defaulting to 168.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at.
    /// For hourly data with weekly seasonality, 168 (one week) is a good start.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the state dimension of the state space model.
    /// </summary>
    /// <value>The state dimension, defaulting to 40.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the hidden state vector.
    /// Larger values can capture more complex dynamics but need more data.
    /// Typically includes components for trend (2-4 dims) and seasonality
    /// (2 dims per seasonal period).
    /// </para>
    /// </remarks>
    public int StateDimension { get; set; } = 40;

    /// <summary>
    /// Gets or sets the hidden dimension of the RNN encoder.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the RNN's internal representation.
    /// The RNN processes the historical data and produces parameters for the SSM.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 50;

    /// <summary>
    /// Gets or sets the number of RNN layers.
    /// </summary>
    /// <value>The number of RNN layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More layers can learn more complex patterns
    /// but are harder to train. 2 layers usually work well.
    /// </para>
    /// </remarks>
    public int NumRnnLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the seasonal periods to model.
    /// </summary>
    /// <value>Array of seasonal periods, defaulting to [24, 168] (daily and weekly for hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The repeating patterns in your data:
    /// - Hourly data: [24, 168] for daily and weekly cycles
    /// - Daily data: [7, 365] for weekly and yearly cycles
    /// - Monthly data: [12] for yearly cycles
    ///
    /// Each period adds 2 dimensions to the state for sin/cos components.
    /// </para>
    /// </remarks>
    public int[] SeasonalPeriods { get; set; } = [24, 168];

    /// <summary>
    /// Gets or sets whether to include a trend component.
    /// </summary>
    /// <value>True to include trend, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trend captures long-term direction (growth or decline).
    /// Disable for stationary data that fluctuates around a constant mean.
    /// </para>
    /// </remarks>
    public bool UseTrend { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include seasonality components.
    /// </summary>
    /// <value>True to include seasonality, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Seasonality captures repeating patterns.
    /// Disable if your data has no periodic patterns.
    /// </para>
    /// </remarks>
    public bool UseSeasonality { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// DeepState often uses lower dropout than pure neural networks.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big of a step to take when updating the model.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to go through the entire training dataset.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples to process at once.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;
}

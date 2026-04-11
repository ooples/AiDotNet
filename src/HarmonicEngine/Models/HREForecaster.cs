using AiDotNet.HarmonicEngine.Options;

namespace AiDotNet.HarmonicEngine.Models;

/// <summary>
/// Time-series forecasting model built on the Harmonic Resonance Engine.
/// Provides windowed input processing, multi-step prediction, and single-pass Hebbian training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This model predicts future values of a time-series (like stock prices)
/// using the HRE's spectral architecture. It works by:
///
/// 1. Taking a window of past values (e.g., the last 64 prices)
/// 2. Processing them through the HRE pipeline (Mellin-Fourier → OFDM → IMD Attention)
/// 3. Outputting a prediction for the next value(s)
///
/// Unlike LSTM or Transformer models that require hundreds of training epochs,
/// the HRE Forecaster can learn from a single pass over the data using Hebbian learning.
/// </para>
/// </remarks>
public class HREForecaster<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly HREModel<T> _model;
    private readonly int _windowSize;
    private readonly int _predictionHorizon;

    /// <summary>
    /// Gets the underlying HRE model.
    /// </summary>
    public HREModel<T> Model => _model;

    /// <summary>
    /// Gets the window size (number of past time steps used as input).
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the prediction horizon (number of future steps to predict).
    /// </summary>
    public int PredictionHorizon => _predictionHorizon;

    /// <summary>
    /// Creates a new HRE Forecaster.
    /// </summary>
    /// <param name="windowSize">Number of past time steps to use as input. Must be a power of 2.</param>
    /// <param name="predictionHorizon">Number of future steps to predict.</param>
    /// <param name="options">HRE model options. InputSize will be set to windowSize.</param>
    public HREForecaster(int windowSize = 64, int predictionHorizon = 1, HREModelOptions? options = null)
    {
        if (windowSize < 2 || (windowSize & (windowSize - 1)) != 0)
            throw new ArgumentException(
                $"Window size must be a power of 2, got {windowSize}.", nameof(windowSize));

        _numOps = MathHelper.GetNumericOperations<T>();
        _windowSize = windowSize;
        _predictionHorizon = predictionHorizon;

        options ??= new HREModelOptions();
        options.InputSize = windowSize;
        options.OutputSize = predictionHorizon;

        _model = new HREModel<T>(options);
    }

    /// <summary>
    /// Predicts the next value(s) given a window of past values.
    /// </summary>
    /// <param name="window">Past values of length WindowSize.</param>
    /// <returns>Predicted values of length PredictionHorizon.</returns>
    public Vector<T> Predict(Vector<T> window)
    {
        if (window.Length != _windowSize)
            throw new ArgumentException(
                $"Window must have {_windowSize} elements, got {window.Length}.");

        var input = new Tensor<T>([_windowSize]);
        for (int i = 0; i < _windowSize; i++)
        {
            input[i] = window[i];
        }

        _model.SetTrainingMode(false);
        var output = _model.Forward(input);

        var prediction = new Vector<T>(_predictionHorizon);
        for (int i = 0; i < _predictionHorizon; i++)
        {
            prediction[i] = output[i];
        }

        return prediction;
    }

    /// <summary>
    /// Performs autoregressive multi-step prediction by feeding predictions back as input.
    /// </summary>
    /// <param name="initialWindow">Starting window of length WindowSize.</param>
    /// <param name="steps">Total number of steps to predict.</param>
    /// <returns>All predicted values.</returns>
    public Vector<T> PredictAutoregressive(Vector<T> initialWindow, int steps)
    {
        if (initialWindow.Length < _windowSize)
            throw new ArgumentException(
                $"Initial window must have at least {_windowSize} elements, got {initialWindow.Length}.",
                nameof(initialWindow));

        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be positive.");

        var predictions = new Vector<T>(steps);
        var currentWindow = new Vector<T>(_windowSize);

        // Copy initial window (last _windowSize elements if longer)
        int offset = initialWindow.Length - _windowSize;
        for (int i = 0; i < _windowSize; i++)
        {
            currentWindow[i] = initialWindow[offset + i];
        }

        int predicted = 0;
        while (predicted < steps)
        {
            var pred = Predict(currentWindow);
            int batchSize = Math.Min(pred.Length, steps - predicted);

            for (int i = 0; i < batchSize; i++)
            {
                predictions[predicted + i] = pred[i];
            }
            predicted += batchSize;

            // Slide window forward
            if (predicted < steps)
            {
                for (int i = 0; i < _windowSize - batchSize; i++)
                {
                    currentWindow[i] = currentWindow[i + batchSize];
                }
                for (int i = 0; i < batchSize; i++)
                {
                    currentWindow[_windowSize - batchSize + i] = pred[i];
                }
            }
        }

        return predictions;
    }

    /// <summary>
    /// Computes the mean squared error between predictions and actual values.
    /// </summary>
    /// <param name="timeSeries">The full time series.</param>
    /// <param name="trainEnd">Index where training data ends and test data begins.</param>
    /// <returns>MSE on the test portion.</returns>
    public double Evaluate(Vector<T> timeSeries, int trainEnd)
    {
        if (timeSeries is null)
            throw new ArgumentNullException(nameof(timeSeries));

        int testLength = timeSeries.Length - trainEnd;
        if (testLength <= 0) return double.NaN;

        double totalError = 0;
        int count = 0;

        for (int t = trainEnd; t < timeSeries.Length - _predictionHorizon; t++)
        {
            // Build window from history
            if (t - _windowSize < 0) continue;

            var window = new Vector<T>(_windowSize);
            for (int i = 0; i < _windowSize; i++)
            {
                window[i] = timeSeries[t - _windowSize + i];
            }

            var pred = Predict(window);

            for (int h = 0; h < _predictionHorizon && t + h < timeSeries.Length; h++)
            {
                double predVal = _numOps.ToDouble(pred[h]);
                double actualVal = _numOps.ToDouble(timeSeries[t + h]);
                double diff = predVal - actualVal;
                totalError += diff * diff;
                count++;
            }
        }

        return count > 0 ? totalError / count : double.NaN;
    }
}

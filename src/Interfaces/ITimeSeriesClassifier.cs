using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for time series classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time series classification assigns labels to entire sequences
/// rather than individual data points. For example, classifying an ECG recording as "normal"
/// or "abnormal", or classifying a gesture based on accelerometer data.</para>
///
/// <para><b>How time series classification differs from regular classification:</b>
/// <list type="bullet">
/// <item>Input is a sequence (time series) not a single feature vector</item>
/// <item>Order of observations matters</item>
/// <item>May have multiple channels (multivariate time series)</item>
/// <item>Sequences can have different lengths</item>
/// </list>
/// </para>
///
/// <para><b>Common Approaches:</b>
/// <list type="bullet">
/// <item><b>Distance-based:</b> DTW + 1-NN, Shapelet-based</item>
/// <item><b>Feature extraction:</b> ROCKET, MiniRocket, TSFresh</item>
/// <item><b>Deep learning:</b> CNN, LSTM, Transformers</item>
/// <item><b>Ensemble:</b> Time Series Forest, BOSS ensemble</item>
/// </list>
/// </para>
///
/// <para><b>Applications:</b>
/// <list type="bullet">
/// <item>Medical diagnosis (ECG, EEG analysis)</item>
/// <item>Human activity recognition (accelerometer data)</item>
/// <item>Speech/audio classification</item>
/// <item>Anomaly detection in sensor data</item>
/// <item>Gesture recognition</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TimeSeriesClassifier")]
public interface ITimeSeriesClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Gets the expected sequence length for input time series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the number of time steps in each input sequence.
    /// Some models require fixed-length input, others can handle variable lengths.</para>
    /// </remarks>
    int SequenceLength { get; }

    /// <summary>
    /// Gets the number of channels (variables) in the time series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For univariate time series, this is 1. For multivariate
    /// (e.g., accelerometer with x, y, z), this equals the number of variables measured.</para>
    /// </remarks>
    int NumChannels { get; }

    /// <summary>
    /// Trains the classifier on a collection of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <param name="labels">Class labels for each sequence.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The sequences tensor holds all training time series stacked
    /// together. Each "row" in the first dimension is one complete time series with its label.</para>
    /// </remarks>
    void TrainOnSequences(Tensor<T> sequences, Vector<T> labels);

    /// <summary>
    /// Predicts class labels for a batch of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <returns>Predicted class labels for each sequence.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass multiple time series at once for efficient batch
    /// prediction. Returns one label per input sequence.</para>
    /// </remarks>
    Vector<T> PredictSequences(Tensor<T> sequences);

    /// <summary>
    /// Predicts class probabilities for a batch of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <returns>Matrix of shape [num_samples, num_classes] with class probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns probabilities for each class for each sequence.
    /// Useful for understanding model confidence and for soft decisions.</para>
    /// </remarks>
    Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences);

    /// <summary>
    /// Gets whether the classifier can handle variable-length sequences.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, sequences don't all need to be the same length.
    /// If false, all sequences must match <see cref="SequenceLength"/>.</para>
    /// </remarks>
    bool SupportsVariableLengths { get; }
}

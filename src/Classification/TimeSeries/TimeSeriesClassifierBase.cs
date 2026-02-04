using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.TimeSeries;

/// <summary>
/// Base class for time series classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for time series
/// classifiers. It handles sequence-based input (3D tensors) and provides the infrastructure
/// for training on time series data while inheriting all the classification machinery from
/// ClassifierBase.</para>
///
/// <para><b>Key concepts:</b>
/// <list type="bullet">
/// <item><b>Sequence:</b> A time-ordered series of observations (e.g., 100 time steps)</item>
/// <item><b>Channel:</b> A variable measured at each time step (e.g., x, y, z accelerometer)</item>
/// <item><b>Sample:</b> One complete time series with its class label</item>
/// </list>
/// </para>
///
/// <para><b>Input format:</b> Sequences are passed as 3D tensors with shape:
/// [num_samples, sequence_length, num_channels]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class TimeSeriesClassifierBase<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>
{
    /// <summary>
    /// Gets or sets the expected sequence length.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of time points in each input sequence.
    /// Set during training based on the training data.</para>
    /// </remarks>
    public int SequenceLength { get; protected set; }

    /// <summary>
    /// Gets or sets the number of channels (variables) in the time series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of measurements at each time point.
    /// For a single sensor, this is 1. For 3-axis accelerometer, this is 3.</para>
    /// </remarks>
    public int NumChannels { get; protected set; }

    /// <summary>
    /// Gets whether this classifier supports variable-length sequences.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Override this property in derived classes that can handle
    /// sequences of different lengths. Default is false (fixed-length required).</para>
    /// </remarks>
    public virtual bool SupportsVariableLengths => false;

    /// <summary>
    /// Gets the time series classifier options.
    /// </summary>
    protected TimeSeriesClassifierOptions<T> TimeSeriesOptions { get; }

    /// <summary>
    /// Creates a new time series classifier base.
    /// </summary>
    /// <param name="options">Configuration options. If null, defaults are used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Options control parameters like sequence length requirements
    /// and channel configuration. Defaults work for most cases.</para>
    /// </remarks>
    protected TimeSeriesClassifierBase(TimeSeriesClassifierOptions<T>? options = null)
        : base(options)
    {
        TimeSeriesOptions = options ?? new TimeSeriesClassifierOptions<T>();
        SequenceLength = TimeSeriesOptions.SequenceLength;
        NumChannels = TimeSeriesOptions.NumChannels;
    }

    /// <summary>
    /// Trains the classifier on a collection of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <param name="labels">Class labels for each sequence.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method learns patterns in the time series data that
    /// distinguish different classes. After training, the model can predict labels for new sequences.</para>
    /// </remarks>
    public virtual void TrainOnSequences(Tensor<T> sequences, Vector<T> labels)
    {
        ValidateSequenceInput(sequences, labels);

        // Extract dimensions from input
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numChan = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;

        SequenceLength = seqLen;
        NumChannels = numChan;

        // Convert 3D sequences to 2D matrix for standard classifier training
        // This flattens each sequence into a single feature vector
        var flattenedData = FlattenSequences(sequences);

        // Call standard training
        Train(flattenedData, labels);
    }

    /// <summary>
    /// Predicts class labels for a batch of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <returns>Predicted class labels for each sequence.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass multiple sequences at once for efficient prediction.
    /// Returns one predicted label per input sequence.</para>
    /// </remarks>
    public virtual Vector<T> PredictSequences(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        // Flatten sequences and use standard prediction
        var flattenedData = FlattenSequences(sequences);
        return Predict(flattenedData);
    }

    /// <summary>
    /// Predicts class probabilities for a batch of time series sequences.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [num_samples, sequence_length, num_channels].</param>
    /// <returns>Matrix of shape [num_samples, num_classes] with class probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns probability scores for each class for each sequence.
    /// Useful for understanding model confidence.</para>
    /// </remarks>
    public virtual Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        // Default implementation returns one-hot encoded predictions
        var predictions = PredictSequences(sequences);
        int numSamples = sequences.Shape[0];
        var probabilities = new Matrix<T>(numSamples, NumClasses);

        for (int i = 0; i < numSamples; i++)
        {
            int classIdx = GetClassIndexFromLabel(predictions[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                probabilities[i, classIdx] = NumOps.One;
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Validates the input sequences have the correct shape and dimensions.
    /// </summary>
    /// <param name="sequences">The input sequences to validate.</param>
    /// <param name="labels">Optional labels to validate (if provided).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures the input data has the expected format before
    /// processing. It catches common errors like wrong dimensions or mismatched lengths.</para>
    /// </remarks>
    protected virtual void ValidateSequenceInput(Tensor<T> sequences, Vector<T>? labels)
    {
        if (sequences is null)
        {
            throw new ArgumentNullException(nameof(sequences));
        }

        if (sequences.Shape.Length < 2)
        {
            throw new ArgumentException("Sequences must be at least 2D [samples, sequence_length].", nameof(sequences));
        }

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numChan = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;

        if (numSamples == 0)
        {
            throw new ArgumentException("No samples provided.", nameof(sequences));
        }

        if (seqLen == 0)
        {
            throw new ArgumentException("Sequence length cannot be zero.", nameof(sequences));
        }

        // Check against expected dimensions if set
        if (SequenceLength > 0 && !SupportsVariableLengths && seqLen != SequenceLength)
        {
            throw new ArgumentException(
                $"Expected sequence length {SequenceLength}, but got {seqLen}. " +
                "This classifier requires fixed-length sequences.",
                nameof(sequences));
        }

        if (NumChannels > 0 && numChan != NumChannels)
        {
            throw new ArgumentException(
                $"Expected {NumChannels} channels, but got {numChan}.",
                nameof(sequences));
        }

        if (labels is not null && labels.Length != numSamples)
        {
            throw new ArgumentException(
                $"Number of labels ({labels.Length}) must match number of samples ({numSamples}).",
                nameof(labels));
        }
    }

    /// <summary>
    /// Flattens 3D sequence data into 2D matrix format for standard classifier processing.
    /// </summary>
    /// <param name="sequences">3D tensor of shape [samples, sequence_length, channels].</param>
    /// <returns>2D matrix of shape [samples, sequence_length * channels].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts the 3D time series data into a 2D matrix where
    /// each row is a flattened sequence. This allows using standard classification algorithms.</para>
    ///
    /// <para><b>Note:</b> Derived classes may override this to use more sophisticated
    /// feature extraction instead of simple flattening.</para>
    /// </remarks>
    protected virtual Matrix<T> FlattenSequences(Tensor<T> sequences)
    {
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numChan = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;

        int flattenedSize = seqLen * numChan;
        var result = new Matrix<T>(numSamples, flattenedSize);

        for (int s = 0; s < numSamples; s++)
        {
            int flatIdx = 0;
            for (int t = 0; t < seqLen; t++)
            {
                for (int c = 0; c < numChan; c++)
                {
                    int[] indices = numChan > 1
                        ? new[] { s, t, c }
                        : new[] { s, t };
                    result[s, flatIdx++] = sequences[indices];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts 2D flattened data back to 3D sequence format.
    /// </summary>
    /// <param name="flattenedData">2D matrix of shape [samples, sequence_length * channels].</param>
    /// <returns>3D tensor of shape [samples, sequence_length, channels].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The reverse of FlattenSequences. Useful for methods that need
    /// to work with the original sequence structure.</para>
    /// </remarks>
    protected virtual Tensor<T> UnflattenToSequences(Matrix<T> flattenedData)
    {
        if (SequenceLength <= 0 || NumChannels <= 0)
        {
            throw new InvalidOperationException("SequenceLength and NumChannels must be set before unflattening.");
        }

        int numSamples = flattenedData.Rows;
        var result = new Tensor<T>(new[] { numSamples, SequenceLength, NumChannels });

        for (int s = 0; s < numSamples; s++)
        {
            int flatIdx = 0;
            for (int t = 0; t < SequenceLength; t++)
            {
                for (int c = 0; c < NumChannels; c++)
                {
                    result[new[] { s, t, c }] = flattenedData[s, flatIdx++];
                }
            }
        }

        return result;
    }
}

using AiDotNet.Classification.Linear;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Classification.TimeSeries;

/// <summary>
/// Implements ROCKET (Random Convolutional Kernel Transform) for time series classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ROCKET is a highly efficient and accurate time series classifier
/// that uses thousands of random convolutional kernels to extract features. Despite using random
/// kernels (no training), it achieves state-of-the-art accuracy while being orders of magnitude
/// faster than other methods.</para>
///
/// <para><b>How ROCKET works:</b>
/// <list type="number">
/// <item>Generate thousands of random convolutional kernels with varying lengths, dilations, and weights</item>
/// <item>Apply each kernel to the input time series to produce an output array</item>
/// <item>Extract two features from each kernel output: max value and proportion of positive values (PPV)</item>
/// <item>Use these features with a simple linear classifier (e.g., Ridge regression)</item>
/// </list>
/// </para>
///
/// <para><b>Why ROCKET is so effective:</b>
/// <list type="bullet">
/// <item>Random kernels + large quantity = covers diverse patterns</item>
/// <item>PPV captures frequency of pattern occurrence</item>
/// <item>Max value captures pattern strength</item>
/// <item>Dilation handles different time scales</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Dempster et al., "ROCKET: Exceptionally fast and accurate time series classification
/// using random convolutional kernels" (2020)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RocketClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>
{
    private readonly List<RocketKernel> _kernels;
    private readonly RocketOptions<T> _rocketOptions;
    private readonly Random _random;
    private RidgeClassifier<T>? _internalClassifier;
    private Vector<T>? _internalWeights;

    /// <summary>
    /// Gets or sets the expected sequence length.
    /// </summary>
    public int SequenceLength { get; protected set; }

    /// <summary>
    /// Gets or sets the number of channels (variables) in the time series.
    /// </summary>
    public int NumChannels { get; protected set; }

    /// <summary>
    /// Gets whether this classifier supports variable-length sequences.
    /// </summary>
    public bool SupportsVariableLengths => false;

    /// <summary>
    /// Represents a single random convolutional kernel used in ROCKET.
    /// </summary>
    private class RocketKernel
    {
        public double[] Weights { get; }
        public int Length { get; }
        public int Dilation { get; }
        public double Bias { get; }
        public int Padding { get; }

        public RocketKernel(double[] weights, int dilation, double bias, int padding)
        {
            Weights = weights;
            Length = weights.Length;
            Dilation = dilation;
            Bias = bias;
            Padding = padding;
        }
    }

    /// <summary>
    /// Creates a new ROCKET classifier.
    /// </summary>
    /// <param name="options">Configuration options for ROCKET.</param>
    public RocketClassifier(RocketOptions<T>? options = null)
        : base(options)
    {
        _rocketOptions = options ?? new RocketOptions<T>();
        _kernels = new List<RocketKernel>(_rocketOptions.NumKernels);
        _random = _rocketOptions.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_rocketOptions.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.TimeSeriesClassifier;

    /// <summary>
    /// Trains the ROCKET classifier on time series data.
    /// </summary>
    public void TrainOnSequences(Tensor<T> sequences, Vector<T> labels)
    {
        ValidateSequenceInput(sequences, labels);

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        NumChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        SequenceLength = seqLen;

        // Generate random kernels if not already done
        if (_kernels.Count == 0)
        {
            GenerateKernels(seqLen);
        }

        // Transform sequences using ROCKET kernels
        var transformedData = TransformSequences(sequences);

        // Train internal Ridge classifier
        _internalClassifier = new RidgeClassifier<T>(new LinearClassifierOptions<T>
        {
            Alpha = 1.0,
            FitIntercept = true
        });

        // For multi-class, use a simple approach - will need OneVsRest for > 2 classes
        ClassLabels = ExtractClassLabels(labels);
        NumClasses = ClassLabels.Length;
        NumFeatures = transformedData.Columns;

        Train(transformedData, labels);
    }

    /// <summary>
    /// Trains the classifier on feature matrix (used after transform).
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Use internal classifier or implement simple logistic regression
        if (NumClasses == 2)
        {
            // Simple binary classification with ridge-like weights
            TrainBinaryClassifier(x, y);
        }
        else
        {
            // Multi-class: Use one-vs-rest approach
            TrainMultiClassClassifier(x, y);
        }
    }

    /// <summary>
    /// Trains a binary classifier using ridge regression approach.
    /// </summary>
    private void TrainBinaryClassifier(Matrix<T> x, Vector<T> y)
    {
        // Convert labels to +1/-1
        T positiveClass = ClassLabels is not null ? ClassLabels[ClassLabels.Length - 1] : NumOps.One;
        var yRegression = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            yRegression[i] = NumOps.Compare(y[i], positiveClass) == 0
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        // Ridge solution: w = (X'X + alpha*I)^(-1) X'y
        double alpha = 1.0;
        var weights = ComputeRidgeWeights(x, yRegression, alpha);
        _internalWeights = weights;
    }

    /// <summary>
    /// Trains a multi-class classifier using one-vs-rest approach.
    /// </summary>
    private void TrainMultiClassClassifier(Matrix<T> x, Vector<T> y)
    {
        // For simplicity, store weights for each class
        // This is a simplified OvR approach
        var allWeights = new List<Vector<T>>();

        for (int c = 0; c < NumClasses; c++)
        {
            var classLabel = ClassLabels is not null ? ClassLabels[c] : NumOps.FromDouble(c);
            var binaryLabels = new Vector<T>(y.Length);

            for (int i = 0; i < y.Length; i++)
            {
                binaryLabels[i] = NumOps.Compare(y[i], classLabel) == 0
                    ? NumOps.One
                    : NumOps.Negate(NumOps.One);
            }

            var classWeights = ComputeRidgeWeights(x, binaryLabels, 1.0);
            allWeights.Add(classWeights);
        }

        // Flatten all weights into single vector
        int totalParams = NumFeatures * NumClasses;
        _internalWeights = new Vector<T>(totalParams);
        int idx = 0;
        foreach (var w in allWeights)
        {
            for (int j = 0; j < w.Length; j++)
            {
                _internalWeights[idx++] = w[j];
            }
        }
    }

    /// <summary>
    /// Computes ridge regression weights.
    /// </summary>
    private Vector<T> ComputeRidgeWeights(Matrix<T> x, Vector<T> y, double alpha)
    {
        int n = x.Columns;

        // Compute X'X + alpha*I
        var xtx = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < x.Rows; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(x[k, i], x[k, j]));
                }
                xtx[i, j] = sum;

                if (i == j)
                {
                    xtx[i, j] = NumOps.Add(xtx[i, j], NumOps.FromDouble(alpha));
                }
            }
        }

        // Compute X'y
        var xty = new Vector<T>(n);
        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < x.Rows; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i, j], y[i]));
            }
            xty[j] = sum;
        }

        // Solve using simple Gauss-Jordan elimination
        return SolveLinearSystem(xtx, xty);
    }

    /// <summary>
    /// Solves a linear system Ax = b using Gaussian elimination.
    /// </summary>
    private Vector<T> SolveLinearSystem(Matrix<T> a, Vector<T> b)
    {
        int n = b.Length;
        var augmented = new Matrix<T>(n, n + 1);

        // Create augmented matrix [A|b]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = a[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++)
        {
            // Find pivot
            int maxRow = k;
            double maxVal = Math.Abs(NumOps.ToDouble(augmented[k, k]));
            for (int i = k + 1; i < n; i++)
            {
                double val = Math.Abs(NumOps.ToDouble(augmented[i, k]));
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = i;
                }
            }

            // Swap rows
            if (maxRow != k)
            {
                for (int j = 0; j <= n; j++)
                {
                    (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
                }
            }

            // Eliminate
            T pivot = augmented[k, k];
            if (Math.Abs(NumOps.ToDouble(pivot)) < 1e-10)
            {
                continue; // Skip near-zero pivot
            }

            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(augmented[i, k], pivot);
                for (int j = k; j <= n; j++)
                {
                    augmented[i, j] = NumOps.Subtract(augmented[i, j], NumOps.Multiply(factor, augmented[k, j]));
                }
            }
        }

        // Back substitution
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Subtract(sum, NumOps.Multiply(augmented[i, j], x[j]));
            }

            T diag = augmented[i, i];
            x[i] = Math.Abs(NumOps.ToDouble(diag)) > 1e-10
                ? NumOps.Divide(sum, diag)
                : NumOps.Zero;
        }

        return x;
    }

    /// <summary>
    /// Predicts class labels for time series sequences.
    /// </summary>
    public Vector<T> PredictSequences(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);
        var transformed = TransformSequences(sequences);
        return Predict(transformed);
    }

    /// <summary>
    /// Predicts class labels for feature matrix.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_internalWeights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            if (NumClasses == 2)
            {
                // Binary prediction
                double score = 0;
                for (int j = 0; j < input.Columns && j < _internalWeights.Length; j++)
                {
                    score += NumOps.ToDouble(input[i, j]) * NumOps.ToDouble(_internalWeights[j]);
                }
                predictions[i] = score >= 0 && ClassLabels is not null
                    ? ClassLabels[ClassLabels.Length - 1]
                    : (ClassLabels is not null ? ClassLabels[0] : NumOps.Zero);
            }
            else
            {
                // Multi-class: find class with highest score
                int bestClass = 0;
                double bestScore = double.MinValue;

                for (int c = 0; c < NumClasses; c++)
                {
                    double score = 0;
                    int weightOffset = c * NumFeatures;
                    for (int j = 0; j < input.Columns && weightOffset + j < _internalWeights.Length; j++)
                    {
                        score += NumOps.ToDouble(input[i, j]) * NumOps.ToDouble(_internalWeights[weightOffset + j]);
                    }

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                predictions[i] = ClassLabels is not null ? ClassLabels[bestClass] : NumOps.FromDouble(bestClass);
            }
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities for time series sequences.
    /// </summary>
    public Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
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

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _internalWeights?.Clone() ?? new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _internalWeights = parameters.Clone();
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new RocketClassifier<T>(_rocketOptions);
        copy._internalWeights = parameters.Clone();
        copy.NumFeatures = NumFeatures;
        copy.NumClasses = NumClasses;
        copy.ClassLabels = ClassLabels?.Clone();
        copy.SequenceLength = SequenceLength;
        copy.NumChannels = NumChannels;

        // Copy kernels
        foreach (var kernel in _kernels)
        {
            copy._kernels.Add(new RocketKernel(
                (double[])kernel.Weights.Clone(),
                kernel.Dilation,
                kernel.Bias,
                kernel.Padding));
        }

        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new RocketClassifier<T>(_rocketOptions);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Ridge regression doesn't use gradients in the traditional sense
        // Return zeros for now
        return new Vector<T>(_internalWeights?.Length ?? 0);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_internalWeights is null) return;

        for (int i = 0; i < _internalWeights.Length && i < gradients.Length; i++)
        {
            _internalWeights[i] = NumOps.Subtract(_internalWeights[i],
                NumOps.Multiply(learningRate, gradients[i]));
        }
    }

    /// <summary>
    /// Validates the input sequences.
    /// </summary>
    private void ValidateSequenceInput(Tensor<T> sequences, Vector<T>? labels)
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

        if (numSamples == 0)
        {
            throw new ArgumentException("No samples provided.", nameof(sequences));
        }

        if (seqLen == 0)
        {
            throw new ArgumentException("Sequence length cannot be zero.", nameof(sequences));
        }

        if (labels is not null && labels.Length != numSamples)
        {
            throw new ArgumentException(
                $"Number of labels ({labels.Length}) must match number of samples ({numSamples}).",
                nameof(labels));
        }
    }

    /// <summary>
    /// Generates the random convolutional kernels.
    /// </summary>
    private void GenerateKernels(int sequenceLength)
    {
        _kernels.Clear();
        int[] kernelLengths = _rocketOptions.KernelLengths ?? [7, 9, 11];

        for (int i = 0; i < _rocketOptions.NumKernels; i++)
        {
            int length = kernelLengths[_random.Next(kernelLengths.Length)];

            var weights = new double[length];
            double sum = 0;
            for (int j = 0; j < length; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                weights[j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                sum += weights[j];
            }

            double mean = sum / length;
            for (int j = 0; j < length; j++)
            {
                weights[j] -= mean;
            }

            int maxDilation = Math.Max(1, (sequenceLength - 1) / (length - 1));
            int dilation = _random.Next(1, maxDilation + 1);
            double bias = _random.NextDouble() * 2 - 1;
            int padding = _random.Next(2);

            _kernels.Add(new RocketKernel(weights, dilation, bias, padding));
        }
    }

    /// <summary>
    /// Transforms sequences into feature vectors using ROCKET kernels.
    /// </summary>
    private Matrix<T> TransformSequences(Tensor<T> sequences)
    {
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        int numFeatures = _kernels.Count * 2 * numChannels;

        var result = new Matrix<T>(numSamples, numFeatures);

        for (int s = 0; s < numSamples; s++)
        {
            int featureIdx = 0;

            for (int c = 0; c < numChannels; c++)
            {
                var channelData = new double[seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    int[] indices = numChannels > 1 ? [s, t, c] : [s, t];
                    channelData[t] = NumOps.ToDouble(sequences[indices]);
                }

                foreach (var kernel in _kernels)
                {
                    var (maxVal, ppv) = ApplyKernel(channelData, kernel);
                    result[s, featureIdx++] = NumOps.FromDouble(maxVal);
                    result[s, featureIdx++] = NumOps.FromDouble(ppv);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies a single kernel and extracts features.
    /// </summary>
    private (double Max, double PPV) ApplyKernel(double[] data, RocketKernel kernel)
    {
        int outputLength = data.Length - (kernel.Length - 1) * kernel.Dilation;
        if (outputLength <= 0)
        {
            return (0, 0);
        }

        double maxVal = double.MinValue;
        int positiveCount = 0;

        for (int i = 0; i < outputLength; i++)
        {
            double sum = 0;
            for (int j = 0; j < kernel.Length; j++)
            {
                int dataIdx = i + j * kernel.Dilation;
                sum += data[dataIdx] * kernel.Weights[j];
            }

            sum -= kernel.Bias;
            maxVal = Math.Max(maxVal, sum);
            if (sum > 0)
            {
                positiveCount++;
            }
        }

        double ppv = (double)positiveCount / outputLength;
        return (maxVal, ppv);
    }
}

/// <summary>
/// Configuration options for ROCKET classifier.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RocketOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of random kernels to generate.
    /// </summary>
    public int NumKernels { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the kernel lengths to use.
    /// </summary>
    public int[]? KernelLengths { get; set; } = [7, 9, 11];

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Validates the options.
    /// </summary>
    public virtual void Validate()
    {
        if (NumKernels < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(NumKernels),
                "Number of kernels must be at least 1.");
        }

        if (KernelLengths is not null && KernelLengths.Length == 0)
        {
            throw new ArgumentException("Kernel lengths cannot be empty.", nameof(KernelLengths));
        }

        if (KernelLengths is not null && KernelLengths.Any(l => l < 3))
        {
            throw new ArgumentException("Kernel lengths must be at least 3.", nameof(KernelLengths));
        }
    }
}

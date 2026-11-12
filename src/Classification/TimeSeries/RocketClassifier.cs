using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification.Linear;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
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
/// <example>
/// <code>
/// // Create ROCKET classifier with short kernel lengths matching the data
/// var options = new RocketOptions&lt;double&gt; { KernelLengths = new[] { 3, 5 } };
///
/// // Prepare time series data: rows are samples, columns are time steps
/// // Series length must be >= max kernel length
/// var features = new Matrix&lt;double&gt;(4, 8);
/// // Class 0: low-amplitude oscillation
/// features[0, 0] = 1.0; features[0, 1] = 1.2; features[0, 2] = 1.5; features[0, 3] = 1.3;
/// features[0, 4] = 1.1; features[0, 5] = 1.4; features[0, 6] = 1.2; features[0, 7] = 1.0;
/// features[1, 0] = 1.1; features[1, 1] = 1.3; features[1, 2] = 1.4; features[1, 3] = 1.2;
/// features[1, 4] = 1.0; features[1, 5] = 1.3; features[1, 6] = 1.1; features[1, 7] = 1.2;
/// // Class 1: high-amplitude trend
/// features[2, 0] = 2.0; features[2, 1] = 2.5; features[2, 2] = 2.3; features[2, 3] = 2.8;
/// features[2, 4] = 3.0; features[2, 5] = 3.2; features[2, 6] = 3.5; features[2, 7] = 3.8;
/// features[3, 0] = 2.1; features[3, 1] = 2.4; features[3, 2] = 2.6; features[3, 3] = 2.9;
/// features[3, 4] = 3.1; features[3, 5] = 3.3; features[3, 6] = 3.4; features[3, 7] = 3.7;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train: extract max and PPV features from random kernels and fit classifier
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RocketClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new time series
/// var newSample = new Matrix&lt;double&gt;(1, 8);
/// newSample[0, 0] = 1.0; newSample[0, 1] = 1.1; newSample[0, 2] = 1.3; newSample[0, 3] = 1.2;
/// newSample[0, 4] = 1.0; newSample[0, 5] = 1.2; newSample[0, 6] = 1.1; newSample[0, 7] = 1.0;
/// var predictions = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Vector<>))]
[ResearchPaper("ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels", "https://arxiv.org/abs/1910.13051", Year = 2020, Authors = "Angus Dempster, Francois Petitjean, Geoffrey I. Webb")]
public partial class RocketClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>{

    private readonly List<RocketKernel> _kernels;
    private readonly RocketOptions<T> _rocketOptions;
    private readonly Random _random;
    [Scratch]
    private RidgeClassifier<T>? _internalClassifier;

    [FittedParameter]
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

    /// <summary>The generated kernels (weights, dilation, bias, padding), in feature order.</summary>
    internal IReadOnlyList<(double[] Weights, int Dilation, double Bias, int Padding)> KernelSpecifications =>
        _kernels.Select(kernel => (kernel.Weights, kernel.Dilation, kernel.Bias, kernel.Padding)).ToList();

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

        // sktime's head: StandardScaler(with_mean=False) + RidgeClassifierCV(alphas=logspace(-3, 3, 10)), one
        // +1/-1 target column for two classes and one per class otherwise.
        _internalWeights = RidgeClassifierCv<T>.Fit(x, y, ClassLabels, RidgeClassifierCv<T>.DefaultAlphas);
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

        return RidgeClassifierCv<T>.Predict(input, _internalWeights, ClassLabels ?? ClassIndexLabels(), NumFeatures);
    }

    private Vector<T> ClassIndexLabels() => AiDotNetEngine.Current.TensorArange<T>(0, NumClasses, 1).ToVector();

    /// <summary>
    /// Predicts class probabilities for time series sequences.
    /// </summary>
    public Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        // Default implementation returns one-hot encoded predictions
        var predictions = PredictSequences(sequences);
        int numSamples = sequences.Shape[0];
        var engine = AiDotNetEngine.Current;
        var shape = new[] { numSamples, NumClasses };
        var classes = (ClassLabels ?? ClassIndexLabels()).AsTensor();
        var oneHot = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(predictions.AsTensor(), new[] { numSamples, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(classes, new[] { 1, NumClasses }), shape));
        var probabilities = oneHot.ToMatrix();

        return probabilities;
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

            // Dempster et al. 2020: dilation = floor(2^x), x ~ U(0, log2((l_input - 1) / (l_kernel - 1))), so the
            // dilations are spread exponentially; bias ~ U(-1, 1); and with probability 1/2 the series is
            // zero-padded by ((l_kernel - 1) * dilation) / 2 at each end so the kernel is centred on every point.
            double maxExponent = Math.Max(0.0, Math.Log((sequenceLength - 1.0) / (length - 1.0), 2.0));
            int dilation = Math.Max(1, (int)Math.Pow(2.0, _random.NextDouble() * maxExponent));
            double bias = _random.NextDouble() * 2 - 1;
            int padding = _random.Next(2) == 1 ? ((length - 1) * dilation) / 2 : 0;

            _kernels.Add(new RocketKernel(weights, dilation, bias, padding));
        }
    }

    /// <summary>
    /// Transforms sequences into feature vectors using ROCKET kernels.
    /// </summary>
    internal Matrix<T> TransformSequences(Tensor<T> sequences)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        int rows = numSamples * numChannels;
        int numKernels = _kernels.Count;

        // One univariate series per (sample, channel), channel-major within a sample: [rows, 1, seqLen].
        var series = numChannels > 1
            ? engine.Reshape(engine.TensorPermute(sequences, new[] { 0, 2, 1 }), new[] { rows, 1, seqLen })
            : engine.Reshape(sequences, new[] { rows, 1, seqLen });

        // Kernels that share a length, dilation and padding run as one multi-channel convolution. Each kernel
        // yields max(W * x + b) and the proportion of positive values (PPV) over its output.
        var features = new Tensor<T>[numKernels];
        var groups = Enumerable.Range(0, numKernels)
            .GroupBy(k => (_kernels[k].Length, _kernels[k].Dilation, _kernels[k].Padding));
        foreach (var group in groups)
        {
            var members = group.ToArray();
            int count = members.Length;
            var (length, dilation, padding) = group.Key;
            int outputLength = seqLen + (2 * padding) - ((length - 1) * dilation);
            Tensor<T> pair;
            if (outputLength <= 0)
            {
                pair = new Tensor<T>(new[] { rows, count, 2 });
            }
            else
            {
                // The kernel weights and biases are fixed tables drawn at generation; converted to T once here.
                var kernelWeights = new Tensor<double>(
                    members.SelectMany(k => _kernels[k].Weights).ToArray(), new[] { count, 1, length }).Cast<T>();
                var biases = new Tensor<double>(members.Select(k => _kernels[k].Bias).ToArray(), new[] { 1, count, 1 }).Cast<T>();
                var outputShape = new[] { rows, count, outputLength };
                var output = engine.TensorAdd(
                    engine.Conv1D(series, kernelWeights, 1, padding, dilation),
                    engine.TensorBroadcastTo(biases, outputShape));
                var max = engine.ReduceMax(output, new[] { 2 }, true);
                var ppv = engine.ReduceMean(engine.TensorGreaterThan(output, NumOps.Zero), new[] { 2 }, true);
                pair = engine.TensorConcatenate(new[] { max, ppv }, 2);
            }

            for (int i = 0; i < count; i++)
            {
                features[members[i]] = engine.TensorSlice(pair, new[] { 0, i, 0 }, new[] { rows, 1, 2 });
            }
        }

        // [rows, kernels, 2] -> [samples, channels * kernels * 2]: channel, then kernel, then (max, PPV).
        var stacked = engine.TensorConcatenate(features, 1);
        return engine.Reshape(stacked, new[] { numSamples, numChannels * numKernels * 2 }).ToMatrix();
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

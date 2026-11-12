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
/// Implements MiniRocket for time series classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MiniRocket is a faster, simpler version of ROCKET that uses
/// deterministic kernels with fixed weights. It achieves similar accuracy to ROCKET while
/// being much faster and more memory efficient.</para>
///
/// <para><b>Key differences from ROCKET:</b>
/// <list type="bullet">
/// <item>Uses only PPV (proportion of positive values) features, not max values</item>
/// <item>Kernels have fixed weights from {-1, 2} (instead of random weights)</item>
/// <item>Uses a fixed set of 84 base kernels (instead of random kernels)</item>
/// <item>Biases are computed from quantiles of convolution outputs</item>
/// </list>
/// </para>
///
/// <para><b>How MiniRocket works:</b>
/// <list type="number">
/// <item>Define 84 deterministic kernel patterns using weights from {-1, 2}</item>
/// <item>For each kernel, compute convolution at multiple dilations</item>
/// <item>Compute bias values from quantiles of convolution outputs</item>
/// <item>Extract PPV features using each (kernel, dilation, bias) combination</item>
/// <item>Train a linear classifier on the extracted features</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic Transform
/// for Time Series Classification" (2021)</para>
/// </remarks>
/// <example>
/// <code>
/// // Create MiniRocket classifier with deterministic kernels for fast time series classification
/// var options = new MiniRocketOptions&lt;double&gt;();
///
/// // Prepare time series data: rows are samples, columns are time steps
/// var features = new Matrix&lt;double&gt;(4, 5);
/// features[0, 0] = 1.0; features[0, 1] = 1.2; features[0, 2] = 1.5; features[0, 3] = 1.3; features[0, 4] = 1.1;
/// features[1, 0] = 1.1; features[1, 1] = 1.3; features[1, 2] = 1.4; features[1, 3] = 1.2; features[1, 4] = 1.0;
/// features[2, 0] = 2.0; features[2, 1] = 2.5; features[2, 2] = 2.3; features[2, 3] = 2.8; features[2, 4] = 3.0;
/// features[3, 0] = 2.1; features[3, 1] = 2.4; features[3, 2] = 2.6; features[3, 3] = 2.9; features[3, 4] = 3.1;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train: extract PPV features using fixed {-1, 2} kernels and fit classifier
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MiniRocketClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new time series
/// var newSample = new Matrix&lt;double&gt;(1, 5);
/// newSample[0, 0] = 1.0; newSample[0, 1] = 1.1; newSample[0, 2] = 1.3; newSample[0, 3] = 1.2; newSample[0, 4] = 1.0;
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
[ResearchPaper("MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification", "https://arxiv.org/abs/2012.08791", Year = 2021, Authors = "Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb")]
public partial class MiniRocketClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    private readonly MiniRocketOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    private int[]? _dilations;
    private int[]? _featuresPerDilation;
    // One [kernels, featuresPerDilation] bias table per dilation, fitted from training convolutions.
    private Tensor<T>[]? _biases;
    [FittedParameter]
    private Vector<T>? _weights;
    private bool _isFitted;

    /// <summary>
    /// Gets the expected sequence length.
    /// </summary>
    public int SequenceLength { get; private set; }

    /// <summary>
    /// Gets the number of channels (variables) in the time series.
    /// </summary>
    public int NumChannels { get; private set; }

    /// <summary>
    /// Gets whether this classifier supports variable-length sequences.
    /// </summary>
    public bool SupportsVariableLengths => false;

    // MiniRocket uses fixed kernel patterns from {-1, 2}
    // These are the 84 unique kernels of length 9 with 3 values equal to 2
    private static readonly int[][] KernelPatterns = GenerateKernelPatterns();

    private const int KernelLength = 9;

    // The 84 kernels as a fixed table of weights {-1, 2}: [kernels, 1, length] for the convolution.
    private static readonly Tensor<T> KernelWeights = new Tensor<int>(
        KernelPatterns.SelectMany(pattern => pattern).ToArray(), new[] { KernelPatterns.Length, 1, KernelLength }).Cast<T>();

    /// <summary>
    /// Creates a new MiniRocket classifier.
    /// </summary>
    /// <param name="options">Configuration options for MiniRocket.</param>
    public MiniRocketClassifier(MiniRocketOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new MiniRocketOptions<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />

    /// <summary>
    /// Trains the MiniRocket classifier on time series sequences.
    /// </summary>
    public void TrainOnSequences(Tensor<T> sequences, Vector<T> labels)
    {
        ValidateSequenceInput(sequences, labels);

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        NumChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        SequenceLength = seqLen;

        // Dilations and the features each one carries
        GenerateKernelsAndDilations(seqLen);

        // Fit biases from training data
        FitBiases(sequences);

        // Transform sequences to features
        var features = Transform(sequences);

        // Train classifier
        ClassLabels = ExtractClassLabels(labels);
        NumClasses = ClassLabels.Length;
        NumFeatures = features.Columns;

        Train(features, labels);
        _isFitted = true;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // sktime's head: StandardScaler(with_mean=False) + RidgeClassifierCV(alphas=logspace(-3, 3, 10)).
        _weights = RidgeClassifierCv<T>.Fit(x, y, ClassLabels, RidgeClassifierCv<T>.DefaultAlphas);
    }

    /// <summary>
    /// Predicts class labels for time series sequences.
    /// </summary>
    public Vector<T> PredictSequences(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);
        var transformed = Transform(sequences);
        return Predict(transformed);
    }

    /// <summary>
    /// Predicts class probabilities for time series sequences.
    /// </summary>
    public Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);
        var predictions = PredictSequences(sequences);
        int numSamples = sequences.Shape[0];
        var engine = AiDotNetEngine.Current;
        var shape = new[] { numSamples, NumClasses };
        var classes = (ClassLabels ?? ClassIndexLabels()).AsTensor();
        var probabilities = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(predictions.AsTensor(), new[] { numSamples, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(classes, new[] { 1, NumClasses }), shape)).ToMatrix();

        return probabilities;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        return RidgeClassifierCv<T>.Predict(input, _weights, ClassLabels ?? ClassIndexLabels(), NumFeatures);
    }

    private Vector<T> ClassIndexLabels() => AiDotNetEngine.Current.TensorArange<T>(0, NumClasses, 1).ToVector();

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = new MiniRocketClassifier<T>(_options);
        clone.SetParameters(parameters);
        clone.ClassLabels = ClassLabels is not null ? new Vector<T>(ClassLabels.ToArray()) : null;
        clone.NumClasses = NumClasses;
        clone.NumFeatures = NumFeatures;
        clone.SequenceLength = SequenceLength;
        clone.NumChannels = NumChannels;
        clone._dilations = _dilations;
        clone._featuresPerDilation = _featuresPerDilation;
        clone._biases = _biases;
        clone._isFitted = _isFitted;
        return clone;
    }

    private static int[][] GenerateKernelPatterns()
    {
        // Generate all 84 combinations of placing 3 values of 2 in a length-9 kernel
        // The rest are -1
        var patterns = new List<int[]>();
        const int kernelLength = 9;

        // Generate all combinations of 3 positions from 9
        for (int i = 0; i < kernelLength - 2; i++)
        {
            for (int j = i + 1; j < kernelLength - 1; j++)
            {
                for (int k = j + 1; k < kernelLength; k++)
                {
                    var pattern = new int[kernelLength];
                    for (int p = 0; p < kernelLength; p++)
                    {
                        pattern[p] = -1;
                    }
                    pattern[i] = 2;
                    pattern[j] = 2;
                    pattern[k] = 2;
                    patterns.Add(pattern);
                }
            }
        }

        return patterns.ToArray();
    }

    /// <summary>
    /// MiniRocket's <c>_fit_dilations</c>: the features per kernel are shared over up to
    /// <see cref="MiniRocketOptions{T}.MaxDilationsPerKernel"/> exponentially spaced dilations
    /// <c>floor(2^x)</c>, <c>x</c> evenly spaced in <c>[0, log2((L - 1) / 8)]</c>; repeated dilations merge and
    /// carry proportionally more features, and the remainder goes round them one at a time.
    /// </summary>
    /// <summary>The fitted dilations and the features each carries per kernel.</summary>
    internal (int[] Dilations, int[] FeaturesPerDilation) DilationPlan =>
        (_dilations ?? Array.Empty<int>(), _featuresPerDilation ?? Array.Empty<int>());

    private void GenerateKernelsAndDilations(int inputLength)
    {
        int numKernels = KernelPatterns.Length;
        int featuresPerKernel = Math.Max(1, _options.NumFeatures / numKernels);
        int dilationCount = Math.Max(1, Math.Min(featuresPerKernel, _options.MaxDilationsPerKernel));
        double multiplier = featuresPerKernel / (double)dilationCount;
        double maxExponent = Math.Max(0.0, Math.Log((inputLength - 1) / (double)(KernelLength - 1), 2.0));

        var grouped = Enumerable.Range(0, dilationCount)
            .Select(i => (int)Math.Pow(2.0, dilationCount == 1 ? 0.0 : maxExponent * i / (dilationCount - 1)))
            .GroupBy(dilation => dilation)
            .OrderBy(group => group.Key)
            .ToArray();
        _dilations = grouped.Select(group => group.Key).ToArray();
        var perDilation = grouped.Select(group => (int)(group.Count() * multiplier)).ToArray();
        int remainder = featuresPerKernel - perDilation.Sum();
        for (int i = 0; remainder > 0; i = (i + 1) % perDilation.Length, remainder--)
        {
            perDilation[i]++;
        }

        _featuresPerDilation = perDilation;
    }

    /// <summary>
    /// MiniRocket's <c>_fit_biases</c>: for each dilation and kernel, one training example drawn at random is
    /// convolved (zero-padded to its own length) and the kernel's biases are that output's quantiles at the
    /// next entries of the low-discrepancy sequence <c>frac(i * golden ratio)</c>.
    /// </summary>
    private void FitBiases(Tensor<T> sequences)
    {
        if (_dilations is null || _featuresPerDilation is null)
        {
            throw new InvalidOperationException("Kernels and dilations must be generated first.");
        }

        var engine = AiDotNetEngine.Current;
        int numExamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numKernels = KernelPatterns.Length;
        var series = UnivariateSeries(sequences);
        double goldenRatio = (Math.Sqrt(5.0) + 1.0) / 2.0;

        _biases = new Tensor<T>[_dilations.Length];
        int quantileIndex = 0;
        for (int d = 0; d < _dilations.Length; d++)
        {
            int dilation = _dilations[d];
            int count = _featuresPerDilation[d];
            int padding = ((KernelLength - 1) * dilation) / 2;

            // One random example per kernel, drawn in kernel order.
            var examples = Enumerable.Range(0, numKernels).Select(_ => _random.Next(numExamples)).ToArray();
            var chosen = engine.TensorGather(series, new Tensor<int>(examples, new[] { numKernels }), 0);
            var (sorted, _) = engine.TensorSort(PerRowConvolution(chosen, dilation, padding), 1, false);
            int outputLength = sorted.Shape[1];

            // numpy's linear quantile: position q (L - 1) between its floor and ceiling. The quantiles are a fixed
            // sequence, so their positions and weights are a constant table converted to T once.
            var quantiles = Enumerable.Range(quantileIndex + 1, numKernels * count)
                .Select(i => (i * goldenRatio) % 1.0).ToArray();
            quantileIndex += numKernels * count;
            var positions = quantiles.Select(q => q * (outputLength - 1)).ToArray();
            var lower = positions.Select(p => (int)Math.Floor(p)).ToArray();
            var upper = lower.Select(i => Math.Min(i + 1, outputLength - 1)).ToArray();
            var fraction = new Tensor<double>(positions.Select((p, i) => p - lower[i]).ToArray(), new[] { numKernels, count }).Cast<T>();
            var low = engine.TensorTakeAlongDim(sorted, new Tensor<int>(lower, new[] { numKernels, count }), 1);
            var high = engine.TensorTakeAlongDim(sorted, new Tensor<int>(upper, new[] { numKernels, count }), 1);
            _biases[d] = engine.TensorAdd(low, engine.TensorMultiply(engine.TensorSubtract(high, low), fraction));
        }
    }

    /// <summary>
    /// MiniRocket's transform: the proportion of each kernel's convolution output above each of its biases
    /// (PPV). Kernels alternate between the zero-padded output and the valid part (padding when
    /// <c>(dilation index + kernel index)</c> is even), so half the features see the series ends.
    /// </summary>
    private Matrix<T> Transform(Tensor<T> sequences)
    {
        if (_dilations is null || _featuresPerDilation is null || _biases is null)
        {
            throw new InvalidOperationException("Model must be fitted before transform.");
        }

        var engine = AiDotNetEngine.Current;
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numKernels = KernelPatterns.Length;
        var series = engine.Reshape(UnivariateSeries(sequences), new[] { numSamples, 1, seqLen });

        var perDilation = new Tensor<T>[_dilations.Length];
        for (int d = 0; d < _dilations.Length; d++)
        {
            int dilation = _dilations[d];
            int count = _featuresPerDilation[d];
            int padding = ((KernelLength - 1) * dilation) / 2;
            var output = engine.Conv1D(series, KernelWeights, 1, padding, dilation);
            int outputLength = output.Shape[2];
            int validLength = outputLength - (2 * padding);

            // Compare in sample chunks so the [samples, kernels, length, biases] comparison stays bounded.
            int chunk = Math.Max(1, 4_000_000 / Math.Max(1, numKernels * outputLength * count));
            var pieces = new List<Tensor<T>>();
            for (int start = 0; start < numSamples; start += chunk)
            {
                int rows = Math.Min(chunk, numSamples - start);
                var cube = new[] { rows, numKernels, outputLength, count };
                var block = engine.TensorSlice(output, new[] { start, 0, 0 }, new[] { rows, numKernels, outputLength });
                var above = engine.TensorGreaterThan(
                    engine.TensorBroadcastTo(engine.Reshape(block, new[] { rows, numKernels, outputLength, 1 }), cube),
                    engine.TensorBroadcastTo(engine.Reshape(_biases[d], new[] { 1, numKernels, 1, count }), cube));
                var full = engine.ReduceMean(above, new[] { 2 }, false);
                var valid = validLength > 0
                    ? engine.ReduceMean(
                        engine.TensorSlice(above, new[] { 0, 0, padding, 0 }, new[] { rows, numKernels, validLength, count }),
                        new[] { 2 }, false)
                    : full;
                var usesPadding = engine.TensorBroadcastTo(PaddedKernelMask(d), new[] { rows, numKernels, count });
                pieces.Add(engine.Reshape(engine.TensorWhere(usesPadding, full, valid), new[] { rows, numKernels * count }));
            }

            perDilation[d] = pieces.Count == 1 ? pieces[0] : engine.TensorConcatenate(pieces.ToArray(), 0);
        }

        // Dilation, then kernel, then bias: MiniRocket's feature order.
        return engine.TensorConcatenate(perDilation, 1).ToMatrix();
    }

    /// <summary>1 for the kernels that use the zero-padded output at dilation index <paramref name="dilationIndex"/>.</summary>
    private static Tensor<T> PaddedKernelMask(int dilationIndex)
    {
        int numKernels = KernelPatterns.Length;
        var flags = Enumerable.Range(0, numKernels).Select(k => (dilationIndex + k) % 2 == 0 ? 1 : 0).ToArray();
        return new Tensor<int>(flags, new[] { 1, numKernels, 1 }).Cast<T>();
    }

    /// <summary>The first channel of every series as [samples, length].</summary>
    private static Tensor<T> UnivariateSeries(Tensor<T> sequences)
    {
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        if (sequences.Shape.Length == 2)
        {
            return sequences;
        }

        var engine = AiDotNetEngine.Current;
        var first = engine.TensorSlice(sequences, new[] { 0, 0, 0 }, new[] { numSamples, seqLen, 1 });
        return engine.Reshape(first, new[] { numSamples, seqLen });
    }

    /// <summary>
    /// Convolves row r of <paramref name="rows"/> with kernel r, zero-padded by <paramref name="padding"/> at each end.
    /// </summary>
    private static Tensor<T> PerRowConvolution(Tensor<T> rows, int dilation, int padding)
    {
        var engine = AiDotNetEngine.Current;
        int count = rows.Shape[0];
        int seqLen = rows.Shape[1];
        int outputLength = seqLen + (2 * padding) - ((KernelLength - 1) * dilation);
        var zeros = new Tensor<T>(new[] { count, padding });
        var padded = padding > 0 ? engine.TensorConcatenate(new[] { zeros, rows, zeros }, 1) : rows;
        var weights = engine.Reshape(KernelWeights, new[] { count, KernelLength });
        var output = new Tensor<T>(new[] { count, outputLength });
        for (int tap = 0; tap < KernelLength; tap++)
        {
            var tapWeight = engine.TensorBroadcastTo(
                engine.TensorSlice(weights, new[] { 0, tap }, new[] { count, 1 }), new[] { count, outputLength });
            var shifted = engine.TensorSlice(padded, new[] { 0, tap * dilation }, new[] { count, outputLength });
            output = engine.TensorAdd(output, engine.TensorMultiply(tapWeight, shifted));
        }

        return output;
    }

    private void ValidateSequenceInput(Tensor<T> sequences, Vector<T>? labels)
    {
        if (sequences is null)
        {
            throw new ArgumentNullException(nameof(sequences));
        }

        if (sequences.Shape.Length < 2)
        {
            throw new ArgumentException("Sequences must be at least 2D [samples, sequence_length].");
        }

        if (_isFitted && sequences.Shape[1] != SequenceLength)
        {
            throw new ArgumentException($"Expected sequence length {SequenceLength}, got {sequences.Shape[1]}.");
        }

        if (labels is not null && labels.Length != sequences.Shape[0])
        {
            throw new ArgumentException("Number of labels must match number of samples.");
        }
    }
}

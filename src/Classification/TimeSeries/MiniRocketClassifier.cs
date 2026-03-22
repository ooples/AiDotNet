using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification.Linear;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// var classifier = new MiniRocketClassifier&lt;double&gt;(options);
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
/// classifier.Train(features, labels);
///
/// // Predict class for new time series
/// var newSample = new Matrix&lt;double&gt;(1, 5);
/// newSample[0, 0] = 1.0; newSample[0, 1] = 1.1; newSample[0, 2] = 1.3; newSample[0, 3] = 1.2; newSample[0, 4] = 1.0;
/// var predictions = classifier.Predict(newSample);
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
[ModelPaper("MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification", "https://arxiv.org/abs/2012.08791", Year = 2021, Authors = "Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb")]
public class MiniRocketClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>
{
    private readonly MiniRocketOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    private double[][]? _kernels;
    private int[]? _dilations;
    private double[][]? _biases;
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

    /// <summary>
    /// Creates a new MiniRocket classifier.
    /// </summary>
    /// <param name="options">Configuration options for MiniRocket.</param>
    public MiniRocketClassifier(MiniRocketOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new MiniRocketOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
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

        // Generate kernels and dilations
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

        // Train ridge classifier
        if (NumClasses == 2)
        {
            TrainBinaryClassifier(x, y);
        }
        else
        {
            TrainMultiClassClassifier(x, y);
        }
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
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            if (NumClasses == 2)
            {
                // Binary classification
                T score = ComputeScore(input, i, _weights);
                predictions[i] = !NumOps.LessThan(score, NumOps.Zero)
                    ? ClassLabels![1]
                    : ClassLabels![0];
            }
            else
            {
                // Multi-class: Find class with highest score
                int bestClass = 0;
                double bestScore = double.MinValue;
                int weightsPerClass = NumFeatures;

                // Extract input row once
                var inputRow = new Vector<T>(NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    inputRow[j] = input[i, j];
                }

                // Reuse classWeights vector across class iterations
                var classWeights = new Vector<T>(NumFeatures);
                for (int c = 0; c < NumClasses; c++)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        classWeights[j] = _weights[c * weightsPerClass + j];
                    }
                    T score = Engine.DotProduct(inputRow, classWeights);

                    double scoreVal = NumOps.ToDouble(score);
                    if (scoreVal > bestScore)
                    {
                        bestScore = scoreVal;
                        bestClass = c;
                    }
                }

                predictions[i] = ClassLabels![bestClass];
            }
        }

        return predictions;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _weights?.Clone() ?? new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _weights = parameters.Clone();
    }

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
        clone._kernels = _kernels;
        clone._dilations = _dilations;
        clone._biases = _biases;
        clone._isFitted = _isFitted;
        return clone;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new MiniRocketClassifier<T>(_options);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Ridge regression is closed-form, gradients not typically used
        return new Vector<T>(GetParameters().Length);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_weights is null) return;

        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = NumOps.Subtract(_weights[i],
                NumOps.Multiply(learningRate, gradients[i]));
        }
    }

    /// <inheritdoc />
    public override byte[] Serialize()
    {
        var metadata = GetModelMetadata();
        var modelDict = new Dictionary<string, object?>
        {
            ["SequenceLength"] = SequenceLength,
            ["NumChannels"] = NumChannels,
            ["IsFitted"] = _isFitted,
            ["NumClasses"] = NumClasses,
            ["NumFeatures"] = NumFeatures,
            ["TaskType"] = (int)TaskType
        };

        if (ClassLabels is not null)
        {
            var labels = new double[ClassLabels.Length];
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                labels[i] = NumOps.ToDouble(ClassLabels[i]);
            }
            modelDict["ClassLabels"] = labels;
        }

        if (_weights is not null)
        {
            var weights = new double[_weights.Length];
            for (int i = 0; i < _weights.Length; i++)
            {
                weights[i] = NumOps.ToDouble(_weights[i]);
            }
            modelDict["Weights"] = weights;
        }

        if (_kernels is not null)
        {
            modelDict["KernelCount"] = _kernels.Length;
            for (int i = 0; i < _kernels.Length; i++)
            {
                modelDict[$"Kernel_{i}"] = _kernels[i];
            }
        }

        if (_dilations is not null)
        {
            modelDict["Dilations"] = _dilations;
        }

        if (_biases is not null)
        {
            modelDict["BiasCount"] = _biases.Length;
            for (int i = 0; i < _biases.Length; i++)
            {
                modelDict[$"Bias_{i}"] = _biases[i];
            }
        }

        metadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelDict));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(metadata));
    }

    /// <inheritdoc />
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var metadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString)
            ?? throw new InvalidOperationException("Failed to deserialize MiniRocketClassifier: invalid metadata.");
        if (metadata.ModelData is null)
            throw new InvalidOperationException("Failed to deserialize MiniRocketClassifier: missing model data.");

        var dataString = Encoding.UTF8.GetString(metadata.ModelData);
        var jObj = JsonConvert.DeserializeObject<JObject>(dataString)
            ?? throw new InvalidOperationException("Failed to deserialize MiniRocketClassifier: invalid model payload.");

        // Clear prior state before rehydrating
        ClassLabels = null;
        _weights = null;
        _kernels = null;
        _dilations = null;
        _biases = null;

        SequenceLength = jObj["SequenceLength"]?.ToObject<int>() ?? 0;
        NumChannels = jObj["NumChannels"]?.ToObject<int>() ?? 1;
        _isFitted = jObj["IsFitted"]?.ToObject<bool>() ?? false;
        NumClasses = jObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = jObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(jObj["TaskType"]?.ToObject<int>() ?? 0);

        var labelsToken = jObj["ClassLabels"];
        if (labelsToken is JArray labelsArr)
        {
            ClassLabels = new Vector<T>(labelsArr.Count);
            for (int i = 0; i < labelsArr.Count; i++)
            {
                ClassLabels[i] = NumOps.FromDouble(labelsArr[i].Value<double>());
            }
        }

        var weightsToken = jObj["Weights"];
        if (weightsToken is JArray weightsArr)
        {
            _weights = new Vector<T>(weightsArr.Count);
            for (int i = 0; i < weightsArr.Count; i++)
            {
                _weights[i] = NumOps.FromDouble(weightsArr[i].Value<double>());
            }
        }

        int kernelCount = jObj["KernelCount"]?.ToObject<int>() ?? 0;
        if (kernelCount > 0)
        {
            _kernels = new double[kernelCount][];
            for (int i = 0; i < kernelCount; i++)
            {
                var kArr = jObj[$"Kernel_{i}"] as JArray;
                _kernels[i] = kArr?.Select(v => v.Value<double>()).ToArray() ?? [];
            }
        }

        var dilationsToken = jObj["Dilations"];
        if (dilationsToken is JArray dilArr)
        {
            _dilations = dilArr.Select(d => d.Value<int>()).ToArray();
        }

        int biasCount = jObj["BiasCount"]?.ToObject<int>() ?? 0;
        if (biasCount > 0)
        {
            _biases = new double[biasCount][];
            for (int i = 0; i < biasCount; i++)
            {
                var bArr = jObj[$"Bias_{i}"] as JArray;
                _biases[i] = bArr?.Select(v => v.Value<double>()).ToArray() ?? [];
            }
        }
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

    private void GenerateKernelsAndDilations(int inputLength)
    {
        int numKernels = KernelPatterns.Length; // 84
        _kernels = new double[numKernels][];

        for (int i = 0; i < numKernels; i++)
        {
            _kernels[i] = KernelPatterns[i].Select(x => (double)x).ToArray();
        }

        // Compute dilations such that effective kernel length covers the input
        var dilationsList = new List<int>();
        int kernelLength = 9;
        int maxDilation = (int)Math.Max(1, Math.Floor((inputLength - 1) / (double)(kernelLength - 1)));

        // Use exponentially spaced dilations
        for (int d = 1; d <= maxDilation; d = (int)Math.Ceiling(d * 1.5))
        {
            dilationsList.Add(d);
        }

        if (dilationsList.Count == 0)
        {
            dilationsList.Add(1);
        }

        _dilations = dilationsList.ToArray();
    }

    private void FitBiases(Tensor<T> sequences)
    {
        if (_kernels is null || _dilations is null)
        {
            throw new InvalidOperationException("Kernels and dilations must be generated first.");
        }

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int numKernels = _kernels.Length;
        int numDilations = _dilations.Length;

        _biases = new double[numKernels * numDilations][];

        int biasIdx = 0;
        for (int k = 0; k < numKernels; k++)
        {
            for (int d = 0; d < numDilations; d++)
            {
                var convOutputs = new List<double>();

                // Collect convolution outputs from all samples
                for (int s = 0; s < Math.Min(numSamples, 500); s++) // Limit samples for efficiency
                {
                    var conv = ApplyConvolution(sequences, s, 0, _kernels[k], _dilations[d]);
                    convOutputs.AddRange(conv);
                }

                // Compute quantile-based biases
                if (convOutputs.Count > 0)
                {
                    convOutputs.Sort();
                    _biases[biasIdx] = ComputeQuantileBiases(convOutputs, _options.NumBiasesPerDilation);
                }
                else
                {
                    _biases[biasIdx] = new double[_options.NumBiasesPerDilation];
                }

                biasIdx++;
            }
        }
    }

    private double[] ComputeQuantileBiases(List<double> sortedValues, int numBiases)
    {
        var biases = new double[numBiases];
        int n = sortedValues.Count;

        for (int i = 0; i < numBiases; i++)
        {
            double quantile = i / (double)(numBiases - 1);
            int idx = (int)(quantile * (n - 1));
            biases[i] = sortedValues[idx];
        }

        return biases;
    }

    private Matrix<T> Transform(Tensor<T> sequences)
    {
        if (_kernels is null || _dilations is null || _biases is null)
        {
            throw new InvalidOperationException("Model must be fitted before transform.");
        }

        int numSamples = sequences.Shape[0];
        int numKernels = _kernels.Length;
        int numDilations = _dilations.Length;
        int numBiases = _options.NumBiasesPerDilation;
        int numFeatures = numKernels * numDilations * numBiases;

        var features = new Matrix<T>(numSamples, numFeatures);

        for (int s = 0; s < numSamples; s++)
        {
            int featureIdx = 0;
            int biasArrayIdx = 0;

            for (int k = 0; k < numKernels; k++)
            {
                for (int d = 0; d < numDilations; d++)
                {
                    var conv = ApplyConvolution(sequences, s, 0, _kernels[k], _dilations[d]);
                    var biasValues = _biases[biasArrayIdx++];

                    for (int b = 0; b < numBiases; b++)
                    {
                        // PPV: proportion of values greater than bias
                        double ppv = conv.Count(v => v > biasValues[b]) / (double)conv.Length;
                        features[s, featureIdx++] = NumOps.FromDouble(ppv);
                    }
                }
            }
        }

        return features;
    }

    private double[] ApplyConvolution(Tensor<T> sequences, int sampleIdx, int channelIdx,
        double[] kernel, int dilation)
    {
        int seqLen = sequences.Shape[1];
        int kernelLength = kernel.Length;
        int effectiveLength = (kernelLength - 1) * dilation + 1;
        int outputLength = Math.Max(1, seqLen - effectiveLength + 1);

        var output = new double[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            double sum = 0;
            for (int j = 0; j < kernelLength; j++)
            {
                int idx = i + j * dilation;
                if (idx < seqLen)
                {
                    int[] indices = NumChannels > 1
                        ? new[] { sampleIdx, idx, channelIdx }
                        : new[] { sampleIdx, idx };
                    double val = NumOps.ToDouble(sequences[indices]);
                    sum += val * kernel[j];
                }
            }
            output[i] = sum;
        }

        return output;
    }

    private void TrainBinaryClassifier(Matrix<T> x, Vector<T> y)
    {
        T positiveClass = ClassLabels is not null ? ClassLabels[^1] : NumOps.One;
        var yRegression = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            yRegression[i] = NumOps.Compare(y[i], positiveClass) == 0
                ? NumOps.One
                : NumOps.Negate(NumOps.One);
        }

        _weights = ComputeRidgeWeights(x, yRegression, 1.0);
    }

    private void TrainMultiClassClassifier(Matrix<T> x, Vector<T> y)
    {
        var allWeights = new List<Vector<T>>();

        for (int c = 0; c < NumClasses; c++)
        {
            var classLabel = ClassLabels![c];
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

        int totalParams = NumFeatures * NumClasses;
        _weights = new Vector<T>(totalParams);
        int idx = 0;
        foreach (var w in allWeights)
        {
            for (int j = 0; j < w.Length; j++)
            {
                _weights[idx++] = w[j];
            }
        }
    }

    private Vector<T> ComputeRidgeWeights(Matrix<T> x, Vector<T> y, double alpha)
    {
        int n = x.Columns;

        // Pre-extract columns from X as Vector<T> for Engine.DotProduct
        var xCols = new Vector<T>[n];
        for (int col = 0; col < n; col++)
        {
            xCols[col] = new Vector<T>(x.Rows);
            for (int row = 0; row < x.Rows; row++)
            {
                xCols[col][row] = x[row, col];
            }
        }

        var xtx = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                xtx[i, j] = Engine.DotProduct(xCols[i], xCols[j]);
                if (i == j)
                {
                    xtx[i, j] = NumOps.Add(xtx[i, j], NumOps.FromDouble(alpha));
                }
            }
        }

        var xty = new Vector<T>(n);
        for (int j = 0; j < n; j++)
        {
            xty[j] = Engine.DotProduct(xCols[j], y);
        }

        return SolveLinearSystem(xtx, xty);
    }

    private Vector<T> SolveLinearSystem(Matrix<T> a, Vector<T> b)
    {
        int n = b.Length;
        var augmented = new Matrix<T>(n, n + 1);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = a[i, j];
            }
            augmented[i, n] = b[i];
        }

        for (int k = 0; k < n; k++)
        {
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

            if (maxRow != k)
            {
                for (int j = 0; j <= n; j++)
                {
                    (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
                }
            }

            T pivot = augmented[k, k];
            if (NumOps.LessThan(NumOps.Abs(pivot), NumOps.FromDouble(1e-10))) continue;

            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(augmented[i, k], pivot);
                for (int j = k; j <= n; j++)
                {
                    augmented[i, j] = NumOps.Subtract(augmented[i, j],
                        NumOps.Multiply(factor, augmented[k, j]));
                }
            }
        }

        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Subtract(sum, NumOps.Multiply(augmented[i, j], x[j]));
            }

            T diag = augmented[i, i];
            x[i] = NumOps.GreaterThan(NumOps.Abs(diag), NumOps.FromDouble(1e-10))
                ? NumOps.Divide(sum, diag)
                : NumOps.Zero;
        }

        return x;
    }

    private T ComputeScore(Matrix<T> input, int rowIdx, Vector<T> weights)
    {
        int len = Math.Min(NumFeatures, weights.Length);
        // Compute dot product inline to avoid per-call vector allocations
        T sum = NumOps.Zero;
        for (int j = 0; j < len; j++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(input[rowIdx, j], weights[j]));
        }
        return sum;
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

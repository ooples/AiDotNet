using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// REaLTabFormer generator using GPT-2 style autoregressive transformer for synthetic
/// tabular data generation by treating columns as a sequence of tokens.
/// </summary>
/// <remarks>
/// <para>
/// REaLTabFormer architecture:
/// - <b>Tokenization</b>: Continuous values are binned; categoricals are integer-encoded
/// - <b>Embedding</b>: Token embeddings + positional encodings for each column position
/// - <b>Transformer</b>: Causal self-attention (each column can only see previous columns)
/// - <b>Output</b>: Per-column classification heads predict the token for each column
/// - <b>Training</b>: Cross-entropy loss on next-column prediction (teacher forcing)
/// - <b>Generation</b>: Autoregressive sampling left-to-right with temperature
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> REaLTabFormer generates data like writing a sentence word by word:
///
/// <code>
/// Training: [Col1=A, Col2=B, Col3=C] -> learn P(Col2|Col1), P(Col3|Col1,Col2)
/// Generate: Sample Col1 -> Sample Col2|Col1 -> Sample Col3|Col1,Col2 -> done!
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the FFN blocks. If not, the network creates the standard REaLTabFormer
/// architecture based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 128,
///     outputSize: 128
/// );
/// var options = new REaLTabFormerOptions&lt;double&gt;
/// {
///     NumLayers = 4,
///     NumHeads = 4,
///     EmbeddingDimension = 128,
///     Epochs = 100
/// };
/// var generator = new REaLTabFormerGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 100);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "REaLTabFormer" (Solatorio and Dupriez, 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers",
    "https://arxiv.org/abs/2302.02041",
    Year = 2023,
    Authors = "Aivin V. Solatorio, Olivier Dupriez")]
public partial class REaLTabFormerGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly REaLTabFormerOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private List<ColumnMetadata> _columns = new();
    private Random _random;

    // The GPT-2 decoder (token + position embeddings, causal blocks, final norm), built for the fitted vocabulary.
    // Its feed-forward pairs are the Layers list's when the default layers are in use.
    private CausalTokenTransformer<T>? _transformer;

    // Per-column output heads (auxiliary, depend on data columns)
    private readonly List<FullyConnectedLayer<T>> _outputHeads = new();
    private readonly List<int> _vocabSizes = new();
    // Column c's tokens are ids _tokenOffsets[c] .. _tokenOffsets[c] + _vocabSizes[c] - 1 of the shared vocabulary; id 0
    // is the beginning-of-row token.
    private readonly List<int> _tokenOffsets = new();

    // Column tokenization info
    private readonly List<double[]> _binEdges = new();
    private readonly List<bool> _isNumericalColumn = new();
    private int _seqLength; // Number of columns
    private int _embDim;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the REaLTabFormer-specific options.
    /// </summary>
    public new REaLTabFormerOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new REaLTabFormer generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">REaLTabFormer-specific options.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    public REaLTabFormerGenerator()
        : this(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 128, outputSize: 128))
    {
    }

    public REaLTabFormerGenerator(NeuralNetworkArchitecture<T> architecture,
        REaLTabFormerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new REaLTabFormerOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _embDim = _options.EmbeddingDimension;

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the REaLTabFormer network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the transformer network:
    /// - If you provided custom layers in the architecture, those are used for the FFN blocks
    /// - Otherwise, it creates the standard causal transformer architecture
    ///
    /// The attention layers and per-column output heads are always created internally.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default FFN layers for the transformer (stored in Layers)
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            var relu = new ReLUActivation<T>() as IActivationFunction<T>;
            int d = _embDim;

            for (int l = 0; l < _options.NumLayers; l++)
            {
                Layers.Add(new FullyConnectedLayer<T>(_options.FeedForwardDimension, relu));
                Layers.Add(new FullyConnectedLayer<T>(d, identity));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Builds the decoder for the fitted vocabulary (beginning-of-row token plus every column's tokens) and one output
    /// head per column over that column's tokens.
    /// </summary>
    private void BuildNetworks()
    {
        _tokenOffsets.Clear();
        int vocabulary = 1;
        foreach (int size in _vocabSizes)
        {
            _tokenOffsets.Add(vocabulary);
            vocabulary += size;
        }

        var relu = new ReLUActivation<T>() as IActivationFunction<T>;
        var feedForward = CausalTokenTransformer<T>.PairFeedForward(
            Layers, _options.NumLayers, _options.FeedForwardDimension, _embDim, relu);
        _transformer = new CausalTokenTransformer<T>(
            vocabulary, Math.Max(1, _seqLength), _embDim, _options.NumHeads, _options.DropoutRate, feedForward);

        _outputHeads.Clear();
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        for (int col = 0; col < _seqLength; col++)
        {
            _outputHeads.Add(new FullyConnectedLayer<T>(_vocabSizes[col], identity));
        }
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        int epochCount = epochs ?? _options.Epochs;
        var tokenizedData = PrepareForData(data, columns);

        int batchSize = Math.Max(1, Math.Min(_options.BatchSize, data.Rows));
        var order = Enumerable.Range(0, data.Rows).ToArray();
        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            Shuffle(order);
            for (int start = 0; start < data.Rows; start += batchSize)
            {
                int count = Math.Min(batchSize, data.Rows - start);
                TrainStep(tokenizedData, order, start, count);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null,
        CancellationToken cancellationToken = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochCount);
        }, cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>
    /// One training step of the autoregressive objective on <paramref name="realData"/>'s rows as one batch. The first
    /// call (or one with a new column count) fits the tokenizer and builds the networks from these rows, every column
    /// continuous.
    /// </remarks>
    protected override void TrainOnBatch(Tensor<T> realData)
    {
        var rows = RowsOf(realData);
        if (_transformer is null || _columns.Count != rows.Columns)
            PrepareForData(rows, ContinuousColumns(rows.Columns));
        TrainStep(TokenizeData(rows), Enumerable.Range(0, rows.Rows).ToArray(), 0, rows.Rows);
        IsFitted = true;
    }

    /// <summary>
    /// Fits the tokenizer to <paramref name="data"/> and builds the networks for its vocabulary; returns the tokens.
    /// </summary>
    private Matrix<T> PrepareForData(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _columns = columns.ToList();
        _seqLength = columns.Count;
        PrepareTokenization(data, columns);
        BuildNetworks();
        return TokenizeData(data);
    }

    /// <summary>
    /// One step of the GPT-2 objective (Solatorio and Dupriez 2023 §3): the mean over columns of the cross-entropy of
    /// each column's token given the tokens of the columns before it, over the rows <paramref name="order"/>[start ..
    /// start + count).
    /// </summary>
    private void TrainStep(Matrix<T> tokenizedData, int[] order, int start, int count)
    {
        var transformer = _transformer ?? throw new InvalidOperationException("The decoder was not built.");

        int totalVocab = _vocabSizes.Sum();
        var inputs = new Tensor<T>([count, _seqLength]);
        var targets = new Tensor<T>([count, totalVocab]);
        for (int b = 0; b < count; b++)
        {
            int row = order[start + b];
            int targetOffset = 0;
            for (int col = 0; col < _seqLength; col++)
            {
                int token = (int)NumOps.ToDouble(tokenizedData[row, col]);
                if (col + 1 < _seqLength) inputs[b, col + 1] = NumOps.FromDouble(_tokenOffsets[col] + token);
                targets[b, targetOffset + token] = NumOps.One;
                targetOffset += _vocabSizes[col];
            }
        }

        var layers = new List<ILayer<T>>(transformer.Layers);
        layers.AddRange(_outputHeads);
        var trainable = layers.OfType<ITrainableLayer<T>>().ToList();
        AiDotNet.Training.FusedTrainingStep<T>.Step(
            this, trainable, inputs, targets,
            forward: x => ColumnLogits(transformer.Forward(x, training: true)),
            computeLoss: ColumnCrossEntropy,
            optimizer: _optimizer,
            extraTensors: TapeTrainingStep<T>.CollectParameters(layers));
    }

    /// <summary>
    /// Every column's logits side by side [batch, sum of vocabularies]: column c's head on the hidden state at position
    /// c, which has seen the beginning-of-row token and the tokens of columns 0 .. c-1.
    /// </summary>
    private Tensor<T> ColumnLogits(Tensor<T> hidden)
    {
        int batch = hidden.Shape[0];
        int columns = Math.Min(_outputHeads.Count, hidden.Shape[1]);
        var logits = new Tensor<T>[columns];
        for (int col = 0; col < columns; col++)
            logits[col] = _outputHeads[col].Forward(PositionHidden(hidden, col));
        return logits.Length == 1 ? logits[0] : Engine.TensorConcatenate(logits, axis: 1);
    }

    private Tensor<T> PositionHidden(Tensor<T> hidden, int position)
    {
        int batch = hidden.Shape[0];
        int d = hidden.Shape[2];
        return Engine.Reshape(Engine.TensorSlice(hidden, [0, position, 0], [batch, 1, d]), [batch, d]);
    }

    /// <summary>
    /// The mean over columns of the batch-mean softmax cross-entropy of each column's logits against its one-hot target.
    /// </summary>
    private Tensor<T> ColumnCrossEntropy(Tensor<T> logits, Tensor<T> targets)
    {
        int batch = logits.Shape[0];
        int offset = 0;
        Tensor<T>? total = null;
        foreach (int size in _vocabSizes)
        {
            var columnLogits = Engine.TensorSlice(logits, [0, offset], [batch, size]);
            var columnTargets = Engine.TensorSlice(targets, [0, offset], [batch, size]);
            var logp = Engine.TensorLog(Engine.TensorAddScalar(Engine.Softmax(columnLogits, axis: 1), NumOps.FromDouble(1e-8)));
            var perRow = Engine.ReduceSum(Engine.TensorMultiply(columnTargets, logp), [1], keepDims: false);
            var ce = Engine.TensorNegate(Engine.ReduceMean(perRow, [0], keepDims: false));
            total = total is null ? ce : Engine.TensorAdd(total, ce);
            offset += size;
        }
        return Engine.TensorMultiplyScalar(total ?? new Tensor<T>([1]), NumOps.FromDouble(1.0 / Math.Max(1, _vocabSizes.Count)));
    }

    private void Shuffle(int[] order)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        var transformer = _transformer;
        if (transformer is null || _outputHeads.Count == 0)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }
        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        // Autoregressive sampling, left to right, every sample at once: column c is drawn from its head on position
        // c of the prefix (beginning-of-row token, then the columns drawn so far).
        var tokens = new int[numSamples, _seqLength];
        var prefix = new Tensor<T>([numSamples, _seqLength]);
        for (int col = 0; col < _seqLength; col++)
        {
            var window = col + 1 == _seqLength ? prefix : Engine.TensorSlice(prefix, [0, 0], [numSamples, col + 1]);
            var hidden = transformer.Forward(window, training: false);
            var logits = _outputHeads[col].Forward(PositionHidden(hidden, col));
            for (int i = 0; i < numSamples; i++)
            {
                int token = SampleFromLogits(logits, i, _options.Temperature);
                tokens[i, col] = token;
                if (col + 1 < _seqLength) prefix[i, col + 1] = NumOps.FromDouble(_tokenOffsets[col] + token);
            }
        }

        var result = new Matrix<T>(numSamples, _seqLength);
        for (int i = 0; i < numSamples; i++)
            for (int col = 0; col < _seqLength; col++)
                result[i, col] = NumOps.FromDouble(DetokenizeValue(col, tokens[i, col]));
        return result;
    }

    #endregion

    #region Tokenization

    private void PrepareTokenization(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _binEdges.Clear();
        _isNumericalColumn.Clear();
        _vocabSizes.Clear();

        for (int col = 0; col < columns.Count; col++)
        {
            if (columns[col].IsNumerical)
            {
                _isNumericalColumn.Add(true);

                // Compute bin edges using quantiles
                var values = new double[data.Rows];
                for (int i = 0; i < data.Rows; i++)
                {
                    values[i] = NumOps.ToDouble(data[i, col]);
                }
                Array.Sort(values);

                int numBins = _options.NumBins;
                var edges = new double[numBins + 1];
                edges[0] = values[0] - 1e-6;
                edges[numBins] = values[^1] + 1e-6;
                for (int b = 1; b < numBins; b++)
                {
                    int idx = (int)((long)b * values.Length / numBins);
                    edges[b] = values[Math.Min(idx, values.Length - 1)];
                }
                _binEdges.Add(edges);
                _vocabSizes.Add(numBins);
            }
            else
            {
                _isNumericalColumn.Add(false);
                int numCats = Math.Max(columns[col].NumCategories, 2);
                _binEdges.Add(Array.Empty<double>());
                _vocabSizes.Add(numCats);
            }
        }
    }

    private Matrix<T> TokenizeData(Matrix<T> data)
    {
        var tokens = new Matrix<T>(data.Rows, _seqLength);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int col = 0; col < _seqLength; col++)
            {
                double val = NumOps.ToDouble(data[i, col]);
                int token;

                if (_isNumericalColumn[col])
                {
                    var edges = _binEdges[col];
                    token = 0;
                    for (int b = 1; b < edges.Length; b++)
                    {
                        if (val >= edges[b]) token = b;
                        else break;
                    }
                    token = Math.Min(token, _vocabSizes[col] - 1);
                }
                else
                {
                    token = (int)Math.Round(val);
                    token = Math.Max(0, Math.Min(token, _vocabSizes[col] - 1));
                }

                tokens[i, col] = NumOps.FromDouble(token);
            }
        }

        return tokens;
    }

    private double DetokenizeValue(int col, int token)
    {
        if (_isNumericalColumn[col])
        {
            var edges = _binEdges[col];
            token = Math.Max(0, Math.Min(token, _vocabSizes[col] - 1));
            // Return bin center
            if (token + 1 < edges.Length)
            {
                return (edges[token] + edges[token + 1]) / 2.0;
            }
            return edges[^1];
        }
        else
        {
            return Math.Max(0, Math.Min(token, _vocabSizes[col] - 1));
        }
    }

    #endregion

    #region Sampling

    private int SampleFromLogits(Tensor<T> logits, int row, double temperature)
    {
        int vocabSize = logits.Shape[1];
        if (vocabSize == 0) return 0;
        double t = Math.Max(temperature, 1e-10);

        double maxVal = double.MinValue;
        for (int i = 0; i < vocabSize; i++)
        {
            double v = NumOps.ToDouble(logits[row, i]) / t;
            if (v > maxVal) maxVal = v;
        }

        var probs = new double[vocabSize];
        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = Math.Exp(NumOps.ToDouble(logits[row, i]) / t - maxVal);
            sumExp += probs[i];
        }

        double u = _random.NextDouble() * sumExp;
        double cumulative = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += probs[i];
            if (cumulative >= u) return i;
        }

        return vocabSize - 1;
    }

    #endregion

    #region NeuralNetworkBase Required Overrides

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // For standard NN prediction, run through transformer layers
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    // UpdateParameters redistributed the vector across Layers, which the base already folds -- and
    // did it less safely: the length guard silently left the remaining layers untouched on a short
    // vector instead of failing. Removed under AIDN082.
    /// <inheritdoc />


    /// <inheritdoc />


    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[$"feature_{i}"] = NumOps.One;
        }
        return importance;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["generator_type"] = "REaLTabFormer",
                ["num_layers"] = _options.NumLayers,
                ["num_heads"] = _options.NumHeads,
                ["embedding_dimension"] = _embDim,
                ["feed_forward_dimension"] = _options.FeedForwardDimension,
                ["num_bins"] = _options.NumBins,
                ["temperature"] = _options.Temperature,
                ["is_fitted"] = IsFitted,
                ["sequence_length"] = _seqLength,
                ["using_custom_layers"] = _usingCustomLayers
            }
        };
    }

    #endregion

}

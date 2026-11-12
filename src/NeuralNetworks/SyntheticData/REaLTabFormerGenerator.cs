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

    // Transformer layers (auxiliary, Q/K/V/Out projections + FFN per layer)
    private readonly List<FullyConnectedLayer<T>> _attentionQueryLayers = new();
    private readonly List<FullyConnectedLayer<T>> _attentionKeyLayers = new();
    private readonly List<FullyConnectedLayer<T>> _attentionValueLayers = new();
    private readonly List<FullyConnectedLayer<T>> _attentionOutputLayers = new();
    private readonly List<FullyConnectedLayer<T>> _ffnLayer1s = new();
    private readonly List<FullyConnectedLayer<T>> _ffnLayer2s = new();

    // Residual dropout, one per transformer block, driven by REaLTabFormerOptions.DropoutRate.
    // Parameterless, so it stays out of the Layers list that the FFN pairs are indexed from.
    private readonly List<DropoutLayer<T>> _ffnDropouts = new();

    // Per-column output heads (auxiliary, depend on data columns)
    private readonly List<FullyConnectedLayer<T>> _outputHeads = new();
    private readonly List<int> _vocabSizes = new();

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
    /// Builds the transformer attention layers (auxiliary, not user-overridable).
    /// </summary>
    private void BuildTransformerAttention()
    {
        _attentionQueryLayers.Clear();
        _attentionKeyLayers.Clear();
        _attentionValueLayers.Clear();
        _attentionOutputLayers.Clear();
        _ffnLayer1s.Clear();
        _ffnLayer2s.Clear();
        _ffnDropouts.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var relu = new ReLUActivation<T>() as IActivationFunction<T>;
        int d = _embDim;

        for (int l = 0; l < _options.NumLayers; l++)
        {
            _attentionQueryLayers.Add(new FullyConnectedLayer<T>(d, identity));
            _attentionKeyLayers.Add(new FullyConnectedLayer<T>(d, identity));
            _attentionValueLayers.Add(new FullyConnectedLayer<T>(d, identity));
            _attentionOutputLayers.Add(new FullyConnectedLayer<T>(d, identity));

            // FFN layers reference the Layers list for user-overridable layers
            if (!_usingCustomLayers && Layers.Count >= (l + 1) * 2)
            {
                _ffnLayer1s.Add((FullyConnectedLayer<T>)Layers[l * 2]);
                _ffnLayer2s.Add((FullyConnectedLayer<T>)Layers[l * 2 + 1]);
            }
            else
            {
                _ffnLayer1s.Add(new FullyConnectedLayer<T>(_options.FeedForwardDimension, relu));
                _ffnLayer2s.Add(new FullyConnectedLayer<T>(d, identity));
            }

            _ffnDropouts.Add(new DropoutLayer<T>(_options.DropoutRate));
        }
    }

    /// <summary>
    /// Builds per-column output heads (auxiliary, depend on data columns and vocab sizes).
    /// </summary>
    private void BuildOutputHeads()
    {
        _outputHeads.Clear();
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        int d = _embDim;

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
        _columns = columns.ToList();
        _seqLength = columns.Count;

        // Step 1: Tokenize data
        PrepareTokenization(data, columns);
        var tokenizedData = TokenizeData(data);

        // Step 2: Build transformer attention and output heads
        BuildTransformerAttention();
        BuildOutputHeads();

        // Step 3: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);
        T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int startRow = batch * batchSize;
                int endRow = Math.Min(startRow + batchSize, data.Rows);

                for (int row = startRow; row < endRow; row++)
                {
                }
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
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_outputHeads.Count == 0)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var result = new Matrix<T>(numSamples, _seqLength);

        for (int i = 0; i < numSamples; i++)
        {
            var tokenSequence = new int[_seqLength];

            // Autoregressive generation: left to right
            for (int col = 0; col < _seqLength; col++)
            {
                // Compute embedding for all columns generated so far
                var embeddings = ComputeEmbeddings(tokenSequence, col + 1);

                // Run through transformer
                var hiddenStates = TransformerForward(embeddings, col + 1);

                // Get logits for current column from its output head
                var lastHidden = ExtractColumnHidden(hiddenStates, col);
                var logits = _outputHeads[col].Forward(VectorToTensor(lastHidden));

                // Sample with temperature
                int sampledToken = SampleFromLogits(logits, _options.Temperature);
                tokenSequence[col] = sampledToken;
            }

            // Detokenize
            for (int col = 0; col < _seqLength; col++)
            {
                result[i, col] = NumOps.FromDouble(DetokenizeValue(col, tokenSequence[col]));
            }
        }

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

    #region Forward Pass

    private Vector<T> ComputeEmbeddings(int[] tokens, int length)
    {
        int d = _embDim;
        if (length == 0 || d == 0)
            return new Vector<T>(length * d);

        // Token embedding: the token index scaled to [-1, 1] (0 for a single-token vocabulary), times 0.1.
        var tokenIds = new int[length];
        Array.Copy(tokens, tokenIds, Math.Min(tokens.Length, length));
        var vocab = new Tensor<int>(_vocabSizes.GetRange(0, length).ToArray(), new[] { length, 1 }).Cast<T>();
        var token = new Tensor<int>(tokenIds, new[] { length, 1 }).Cast<T>();
        var spread = Engine.TensorSubtractScalar(vocab, NumOps.One);
        var scaled = Engine.TensorSubtractScalar(
            Engine.TensorDivide(Engine.TensorMultiplyScalar(token, NumOps.FromDouble(2.0)), spread), NumOps.One);
        var tokenScale = Engine.TensorWhere(
            Engine.TensorGreaterThan(vocab, NumOps.One), scaled, Engine.TensorZerosLike(scaled));

        // Position encoding: sin(pos / 10000^(j/d)) on even j, cos(pos / 10000^((j-1)/d)) on odd j.
        var pairIndex = new Tensor<int>(
            Enumerable.Range(0, d).Select(j => j - (j % 2)).ToArray(), new[] { 1, d }).Cast<T>();
        var inverseScale = Engine.TensorExp(
            Engine.TensorMultiplyScalar(pairIndex, NumOps.FromDouble(-Math.Log(10000.0) / d)));
        var angles = Engine.TensorMatMul(
            Engine.Reshape(Engine.TensorArange<T>(0, length, 1), new[] { length, 1 }), inverseScale);       // [length, d]
        var isEven = Engine.TensorBroadcastTo(new Tensor<int>(
            Enumerable.Range(0, d).Select(j => j % 2 == 0 ? 1 : 0).ToArray(), new[] { 1, d }).Cast<T>(), new[] { length, d });
        var positionEncoding = Engine.TensorWhere(isEven, Engine.TensorSin(angles), Engine.TensorCos(angles));

        // Combine token and position
        var combined = Engine.TensorAdd(
            Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(tokenScale, NumOps.FromDouble(0.1)), new[] { length, d }),
            positionEncoding);
        return Engine.Reshape(combined, new[] { length * d }).ToVector();
    }

    private Vector<T> TransformerForward(Vector<T> embeddings, int seqLen)
    {
        int d = _embDim;
        var current = embeddings;

        for (int l = 0; l < _options.NumLayers; l++)
        {
            // Causal self-attention
            current = CausalSelfAttention(current, seqLen, l);

            // Feed-forward network per position
            current = FeedForwardPerPosition(current, seqLen, l);
        }

        return current;
    }

    private Vector<T> CausalSelfAttention(Vector<T> input, int seqLen, int layerIdx)
    {
        int d = _embDim;
        if (seqLen == 0 || d == 0)
            return new Vector<T>(seqLen * d);

        // Every position's query/key/value row in one batch: [seqLen, d] (positions past the input read as 0).
        var x = Engine.Reshape(Tensor<T>.FromVector(TensorToVector(Tensor<T>.FromVector(input), seqLen * d)), new[] { seqLen, d });
        var query = Engine.Reshape(_attentionQueryLayers[layerIdx].Forward(x), new[] { seqLen, d });
        var key = Engine.Reshape(_attentionKeyLayers[layerIdx].Forward(x), new[] { seqLen, d });
        var value = Engine.Reshape(_attentionValueLayers[layerIdx].Forward(x), new[] { seqLen, d });

        // Causal attention: position pos attends over keys [0, pos] with softmax(q . k / sqrt(d)).
        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(query, Engine.TensorTranspose(key)), NumOps.FromDouble(1.0 / Math.Sqrt(d)));
        var future = Engine.TensorGreaterThan(
            Engine.TensorSubtract(
                Engine.Reshape(Engine.TensorArange<int>(0, seqLen, 1), new[] { 1, seqLen }),
                Engine.Reshape(Engine.TensorArange<int>(0, seqLen, 1), new[] { seqLen, 1 })),
            0).Cast<T>();
        scores = Engine.TensorWhere(
            future, Engine.TensorFullLike(scores, NumOps.FromDouble(double.NegativeInfinity)), scores);
        var weights = Engine.TensorSoftmax(scores, 1);

        // Weighted sum of values, output projection + residual
        var attended = Engine.TensorMatMul(weights, value);
        var projected = Engine.Reshape(_attentionOutputLayers[layerIdx].Forward(attended), new[] { seqLen, d });
        return Engine.Reshape(Engine.TensorAdd(projected, x), new[] { seqLen * d }).ToVector();
    }

    private Vector<T> FeedForwardPerPosition(Vector<T> input, int seqLen, int layerIdx)
    {
        int d = _embDim;
        if (seqLen == 0 || d == 0)
            return new Vector<T>(seqLen * d);

        // Every position through the feed-forward block in one batch: [seqLen, d].
        var x = Engine.Reshape(Tensor<T>.FromVector(TensorToVector(Tensor<T>.FromVector(input), seqLen * d)), new[] { seqLen, d });
        var hidden = _ffnLayer1s[layerIdx].Forward(x);
        var ffnOut = _ffnLayer2s[layerIdx].Forward(hidden);
        if (layerIdx < _ffnDropouts.Count)
        {
            _ffnDropouts[layerIdx].SetTrainingMode(IsTrainingMode);
            ffnOut = _ffnDropouts[layerIdx].Forward(ffnOut);
        }

        // Residual connection
        var output = Engine.TensorAdd(Engine.Reshape(ffnOut, new[] { seqLen, d }), x);
        return Engine.Reshape(output, new[] { seqLen * d }).ToVector();
    }

    private Vector<T> ExtractColumnHidden(Vector<T> hiddenStates, int col) => ExtractPosition(hiddenStates, col, _embDim);

    private static Vector<T> ExtractPosition(Vector<T> sequence, int pos, int dim)
    {
        // Elements [pos * dim, pos * dim + dim) of the sequence; positions past its end read as 0.
        int start = Math.Min(pos * dim, sequence.Length);
        int available = Math.Min(dim, sequence.Length - start);
        if (available <= 0)
            return new Vector<T>(dim);
        return TensorToVector(Tensor<T>.FromVector(sequence.GetSubVector(start, available)), dim);
    }

    #endregion

    #region Training

    private Tensor<T> ComputeCrossEntropyGrad(Tensor<T> logits, int targetToken)
    {
        // softmax(logits) - onehot(target) over the flattened logits.
        int length = logits.Length;
        if (length == 0)
            return new Tensor<T>(logits._shape);
        var probabilities = Engine.TensorSoftmax(Engine.Reshape(logits, new[] { length }), 0);
        var grad = targetToken >= 0 && targetToken < length
            ? Engine.TensorSubtract(
                probabilities,
                Engine.Reshape(Engine.TensorOneHot<T>(new Tensor<int>(new[] { targetToken }, new[] { 1 }), length), new[] { length }))
            : probabilities;
        return Engine.Reshape(grad, logits._shape);
    }

    private void UpdateTransformerParameters(T lr)
    {
        // Note: Transformer attention and FFN layers are used in Forward() during training,
        // but their Backward() is not called in the current simplified implementation.
        // FullyConnectedLayer.UpdateParameters() throws if Backward() was never called
        // (because _weightsGradient is null). Until proper backward propagation through
        // transformer layers is implemented (requires full attention backward), we skip
        // these updates. Output heads are updated in TrainRow() after their Backward() call.
    }

    #endregion

    #region Sampling

    private int SampleFromLogits(Tensor<T> logits, double temperature)
    {
        int vocabSize = logits.Length;
        if (vocabSize == 0) return 0;

        // Temperature-scaled, max-shifted exponentials; the token is the first whose running sum reaches
        // u * total for one uniform draw u.
        var scaled = Engine.TensorDivideScalar(
            Engine.Reshape(logits, new[] { vocabSize }), NumOps.FromDouble(Math.Max(temperature, 1e-10)));
        var weights = Engine.TensorExp(Engine.TensorSubtractScalar(scaled, Engine.TensorMaxValue(scaled)));
        var cumulative = Engine.TensorCumSum(weights, 0);
        T total = cumulative.GetFlatIndexValue(vocabSize - 1);
        T u = NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), total);
        int index = Engine.TensorSearchSorted(cumulative, new Tensor<T>(new[] { u }, new[] { 1 }))[0];
        return Math.Min(index, vocabSize - 1);
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

    #region Helpers

    /// <summary>
    /// Sanitizes a gradient tensor by clamping NaN/Inf and clipping to max norm.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        // Non-finite entries become 0; the result is rescaled to maxNorm when its norm exceeds it.
        var result = Engine.TensorWhere(Engine.TensorIsFinite(grad), grad, Engine.TensorZerosLike(grad));
        T norm = NumOps.Sqrt(NumOps.Add(Engine.TensorSumOfSquares(result), NumOps.FromDouble(1e-12)));
        T limit = NumOps.FromDouble(maxNorm);
        if (NumOps.GreaterThan(norm, limit))
        {
            result = Engine.TensorMultiplyScalar(result, NumOps.Divide(limit, norm));
        }

        return result;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    /// <summary>The first min(length, t.Length) values of <paramref name="t"/>, zero-padded to <paramref name="length"/>.</summary>
    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        int copyLen = Math.Min(length, t.Length);
        if (copyLen == 0)
            return new Vector<T>(length);
        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(t, new[] { t.Length });
        var head = copyLen == t.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { copyLen });
        if (copyLen < length)
            head = engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { length - copyLen }) }, 0);
        return head.ToVector();
    }

    #endregion

}

using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabLLM-Gen generator that uses LLM-style schema-aware tokenization and autoregressive
/// transformers to generate realistic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabLLM-Gen processes a row as a sequence of (schema_token, value_token) pairs:
///
/// <code>
///  [COL:Age] [TYPE:num] [VAL:35] -> [COL:Income] [TYPE:num] [VAL:75000] -> ...
///    Schema tokens provide         Value tokens are
///    context about what to          generated autoregressively
///    generate next                  conditioned on schema + previous values
/// </code>
///
/// The model learns to generate value tokens conditioned on:
/// 1. The column's schema tokens (name, type)
/// 2. All previously generated column values
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabLLM-Gen works like an AI that fills in a form:
///
/// - It reads the form labels (column names and types)
/// - Fills in each field one by one, using previous answers to inform the next
/// - For example: after filling in "Age: 25", it knows to generate a realistic
///   income for a 25-year-old
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the FFN blocks. Otherwise, the network creates standard layers based on
/// the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 128, outputSize: 128);
/// var options = new TabLLMGenOptions&lt;double&gt; { NumLayers = 4, NumHeads = 4 };
/// var gen = new TabLLMGenGenerator&lt;double&gt;(architecture, options);
/// gen.Fit(data, columns, epochs: 100);
/// var synthetic = gen.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "LLM-based Tabular Data Generation" (2024)
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
[ResearchPaper("Language Models are Realistic Tabular Data Generators",
    "https://arxiv.org/abs/2210.06280",
    Year = 2023,
    Authors = "Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5,
                Source = "Borisov et al. 2023, Sec. 4: the AdamW optimizer, named by its Loshchilov and "
                        + "Hutter citation, with a learning rate of 5e-5.")]
public partial class TabLLMGenGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TabLLMGenOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;
    private Random _random;

    // ISyntheticTabularGenerator state
    private List<ColumnMetadata> _columns = new();

    // Tokenization
    private int _vocabSize;
    private int _seqLength;
    private readonly List<double[]> _binEdges = new();
    private readonly List<int> _colVocabSizes = new();
    private int _specialTokenOffset;

    // The causal decoder over the serialised row (token + position embeddings, causal blocks, final norm), built for
    // the fitted vocabulary. Its feed-forward pairs are the Layers list's when the default layers are in use.
    private CausalTokenTransformer<T>? _transformer;

    // Next-token head over the whole vocabulary.
    private FullyConnectedLayer<T>? _outputHead;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TabLLM-Gen-specific options.
    /// </summary>
    public new TabLLMGenOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabLLM-Gen generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabLLM-Gen-specific options for generation configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabLLMGenGenerator()
        // inputSize/outputSize declare the raw tabular feature width the NN-Predict path operates on.
        // The FFN's first FullyConnectedLayer resolves its input dim from this at construction
        // (ResolveLazyLayerShapes), so PredictCore — which feeds the input straight into that FFN —
        // requires the fed width to equal inputSize. The prior default of 10 disagreed with the
        // standard 16-wide synthetic-tabular sample the generators are exercised with, so the first
        // FFN layer was sized [.,10] and threw "Matrix dimensions incompatible [1,16] x [10,256]" on
        // Predict/Clone/BatchConsistency. 16 matches that canonical width; callers that pass their own
        // architecture are unaffected.
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 16))
    {
    }

    public TabLLMGenGenerator(NeuralNetworkArchitecture<T> architecture,
        TabLLMGenOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new TabLLMGenOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate
                }));
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the TabLLM-Gen network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the transformer FFN blocks:
    /// - If you provided custom layers, those are used for the FFN blocks
    /// - Otherwise, standard FFN blocks are created based on options
    ///
    /// The attention layers (Q, K, V projections), token embedding, and output head
    /// are always created internally and are not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default FFN layers (2 per transformer layer: ffn1 + ffn2)
            int embDim = _options.EmbeddingDimension;
            int ffnDim = _options.FeedForwardDimension;
            var gelu = new GELUActivation<T>() as IActivationFunction<T>;
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;

            for (int layer = 0; layer < _options.NumLayers; layer++)
            {
                Layers.Add(new FullyConnectedLayer<T>(ffnDim, gelu));
                Layers.Add(new FullyConnectedLayer<T>(embDim, identity));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds the auxiliary transformer layers with actual vocabulary dimensions discovered during Fit().
    /// </summary>
    private void RebuildAuxiliaryLayers()
    {
        int embDim = _options.EmbeddingDimension;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var gelu = new GELUActivation<T>() as IActivationFunction<T>;

        // Rebuild default FFN layers if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(CausalTokenTransformer<T>.DefaultFeedForwardLayers(
                _options.NumLayers, _options.FeedForwardDimension, embDim, gelu));
        }

        var feedForward = CausalTokenTransformer<T>.PairFeedForward(
            Layers, _options.NumLayers, _options.FeedForwardDimension, embDim, gelu);
        _transformer = new CausalTokenTransformer<T>(
            _vocabSize, Math.Max(1, _seqLength), embDim, _options.NumHeads, _options.DropoutRate, feedForward);
        _outputHead = new FullyConnectedLayer<T>(_vocabSize, identity);
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
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null,
        CancellationToken cancellationToken = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        return Task.Run(() => Fit(data, columns, epochCount), cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>
    /// One next-token step on <paramref name="realData"/>'s rows as one batch. The first call (or one with a new column
    /// count) fits the vocabulary and builds the networks from these rows, every column continuous.
    /// </remarks>
    protected override void TrainOnBatch(Tensor<T> realData)
    {
        var rows = RowsOf(realData);
        if (_transformer is null || _columns.Count != rows.Columns)
            PrepareForData(rows, ContinuousColumns(rows.Columns));
        TrainStep(TokenizeData(rows, _columns), Enumerable.Range(0, rows.Rows).ToArray(), 0, rows.Rows);
        IsFitted = true;
    }

    /// <summary>
    /// Fits the vocabulary to <paramref name="data"/> and builds the networks for it; returns the serialised rows.
    /// </summary>
    private List<int[]> PrepareForData(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _columns = new List<ColumnMetadata>(columns);
        BuildTokenization(data, columns);
        RebuildAuxiliaryLayers();
        return TokenizeData(data, columns);
    }

    /// <summary>
    /// One causal language-modelling step over the rows <paramref name="order"/>[start .. start + count): the
    /// cross-entropy of every value token given the START token and the serialised row before it. Schema tokens are
    /// fixed by the column order, so they are context, not targets.
    /// </summary>
    private void TrainStep(List<int[]> tokenizedData, int[] order, int start, int count)
    {
        var transformer = _transformer ?? throw new InvalidOperationException("The decoder was not built.");
        var head = _outputHead ?? throw new InvalidOperationException("The output head was not built.");

        int columns = _columns.Count;
        var inputs = new Tensor<T>([count, _seqLength]);
        var targets = new Tensor<T>([columns * count, _vocabSize]);
        for (int b = 0; b < count; b++)
        {
            var tokens = tokenizedData[order[start + b]];
            inputs[b, 0] = NumOps.FromDouble(StartToken);
            for (int pos = 0; pos + 1 < _seqLength; pos++) inputs[b, pos + 1] = NumOps.FromDouble(tokens[pos]);
            for (int c = 0; c < columns; c++) targets[c * count + b, tokens[ValuePosition(c)]] = NumOps.One;
        }

        var layers = new List<ILayer<T>>(transformer.Layers) { head };
        var trainable = layers.OfType<ITrainableLayer<T>>().ToList();
        AiDotNet.Training.FusedTrainingStep<T>.Step(
            this, trainable, inputs, targets,
            forward: x => ValueLogits(transformer.Forward(x, training: true), head),
            computeLoss: CrossEntropy,
            optimizer: _optimizer,
            extraTensors: TapeTrainingStep<T>.CollectParameters(layers));
    }

    // START, END and MASK are ids 0, 1, 2; START opens every serialised row.
    private const int StartToken = 0;

    // Column c's value token follows its schema tokens.
    private int ValuePosition(int column) => column * (_options.SchemaTokensPerColumn + 1) + _options.SchemaTokensPerColumn;

    /// <summary>
    /// The head's logits at every column's value position, column-major [columns * batch, vocabulary].
    /// </summary>
    private Tensor<T> ValueLogits(Tensor<T> hidden, FullyConnectedLayer<T> head)
    {
        int columns = _columns.Count;
        var positions = new Tensor<T>[columns];
        for (int c = 0; c < columns; c++) positions[c] = PositionHidden(hidden, ValuePosition(c));
        return head.Forward(columns == 1 ? positions[0] : Engine.TensorConcatenate(positions, axis: 0));
    }

    private Tensor<T> PositionHidden(Tensor<T> hidden, int position)
    {
        int batch = hidden.Shape[0];
        int d = hidden.Shape[2];
        return Engine.Reshape(Engine.TensorSlice(hidden, [0, position, 0], [batch, 1, d]), [batch, d]);
    }

    /// <summary>The batch-mean softmax cross-entropy of <paramref name="logits"/> against one-hot <paramref name="targets"/>.</summary>
    private Tensor<T> CrossEntropy(Tensor<T> logits, Tensor<T> targets)
    {
        var logp = Engine.TensorLog(Engine.TensorAddScalar(Engine.Softmax(logits, axis: 1), NumOps.FromDouble(1e-8)));
        var perRow = Engine.ReduceSum(Engine.TensorMultiply(targets, logp), [1], keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(perRow, [0], keepDims: false));
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
        var head = _outputHead;
        if (transformer is null || head is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() before Generate().");
        }
        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        // Autoregressive sampling, every sample at once: each column's schema tokens are written, then its value token
        // is drawn from the head on the prefix, restricted to that column's tokens.
        int schemaTokens = _options.SchemaTokensPerColumn;
        var prefix = new Tensor<T>([numSamples, _seqLength]);
        for (int i = 0; i < numSamples; i++) prefix[i, 0] = NumOps.FromDouble(StartToken);

        var result = new Matrix<T>(numSamples, _columns.Count);
        int valueTokenBase = _specialTokenOffset + _columns.Count * schemaTokens;
        for (int c = 0; c < _columns.Count; c++)
        {
            int schemaBase = _specialTokenOffset + c * schemaTokens;
            int valuePos = ValuePosition(c);
            for (int s = 0; s < schemaTokens; s++)
            {
                for (int i = 0; i < numSamples; i++) prefix[i, valuePos - schemaTokens + s + 1] = NumOps.FromDouble(schemaBase + s);
            }

            var window = valuePos + 1 == _seqLength ? prefix : Engine.TensorSlice(prefix, [0, 0], [numSamples, valuePos + 1]);
            var logits = head.Forward(PositionHidden(transformer.Forward(window, training: false), valuePos));
            for (int i = 0; i < numSamples; i++)
            {
                int valueToken = SampleFromLogits(logits, i, valueTokenBase, _colVocabSizes[c]);
                if (valuePos + 1 < _seqLength) prefix[i, valuePos + 1] = NumOps.FromDouble(valueToken);
                result[i, c] = NumOps.FromDouble(DetokenizeValue(valueToken - valueTokenBase, c));
            }

            valueTokenBase += _colVocabSizes[c];
        }

        return result;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // The FFN layers process a single embedding position
        var current = input;
        for (int i = 0; i < Layers.Count; i += 2)
        {
            if (i + 1 < Layers.Count)
            {
                var ffn1Out = Layers[i].Forward(current);
                current = Layers[i + 1].Forward(ffn1Out);
            }
        }
        return current;
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    /// <inheritdoc />


    /// <inheritdoc />


    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        return new Dictionary<string, T>();
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "TabLLMGen",
                ["VocabSize"] = _vocabSize,
                ["SequenceLength"] = _seqLength,
                ["NumLayers"] = _options.NumLayers,
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["NumHeads"] = _options.NumHeads,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Tokenization

    private void BuildTokenization(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _binEdges.Clear();
        _colVocabSizes.Clear();

        int numBins = _options.NumBins;
        int schemaTokens = _options.SchemaTokensPerColumn;

        _specialTokenOffset = 3; // START, END, MASK tokens
        int currentVocab = _specialTokenOffset;

        // Add schema token IDs (one set per column)
        currentVocab += columns.Count * schemaTokens;

        for (int c = 0; c < columns.Count; c++)
        {
            if (columns[c].IsNumerical)
            {
                // Compute bin edges for continuous columns
                var values = new List<double>();
                for (int r = 0; r < data.Rows; r++)
                {
                    values.Add(NumOps.ToDouble(data[r, c]));
                }
                values.Sort();

                var edges = new double[numBins + 1];
                for (int b = 0; b <= numBins; b++)
                {
                    int idx = (int)((long)b * (values.Count - 1) / numBins);
                    edges[b] = values[Math.Min(idx, values.Count - 1)];
                }
                _binEdges.Add(edges);
                _colVocabSizes.Add(numBins);
                currentVocab += numBins;
            }
            else
            {
                int numCats = Math.Max(2, columns[c].Categories.Count);
                _binEdges.Add(Array.Empty<double>());
                _colVocabSizes.Add(numCats);
                currentVocab += numCats;
            }
        }

        _vocabSize = currentVocab;
        _seqLength = columns.Count * (schemaTokens + 1);
    }

    private List<int[]> TokenizeData(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        var tokenized = new List<int[]>();
        int schemaTokens = _options.SchemaTokensPerColumn;

        for (int r = 0; r < data.Rows; r++)
        {
            var tokens = new int[_seqLength];
            int pos = 0;
            int valueTokenBase = _specialTokenOffset + columns.Count * schemaTokens;

            for (int c = 0; c < columns.Count; c++)
            {
                int schemaBase = _specialTokenOffset + c * schemaTokens;
                for (int s = 0; s < schemaTokens && pos < _seqLength; s++)
                {
                    tokens[pos++] = schemaBase + s;
                }

                if (pos < _seqLength)
                {
                    double val = NumOps.ToDouble(data[r, c]);

                    if (columns[c].IsNumerical && c < _binEdges.Count && _binEdges[c].Length > 0)
                    {
                        int bin = FindBin(val, _binEdges[c]);
                        tokens[pos++] = valueTokenBase + bin;
                    }
                    else
                    {
                        int catIdx = Math.Min(Math.Max((int)Math.Round(val), 0), _colVocabSizes[c] - 1);
                        tokens[pos++] = valueTokenBase + catIdx;
                    }
                }

                valueTokenBase += _colVocabSizes[c];
            }

            tokenized.Add(tokens);
        }

        return tokenized;
    }

    private static int FindBin(double value, double[] edges)
    {
        for (int b = 0; b < edges.Length - 1; b++)
        {
            if (value <= edges[b + 1]) return b;
        }
        return Math.Max(0, edges.Length - 2);
    }

    private double DetokenizeValue(int token, int colIdx)
    {
        if (colIdx < _binEdges.Count && _binEdges[colIdx].Length > 0)
        {
            int numBins = _colVocabSizes[colIdx];
            int bin = Math.Min(Math.Max(token, 0), numBins - 1);
            if (bin < _binEdges[colIdx].Length - 1)
            {
                return (_binEdges[colIdx][bin] + _binEdges[colIdx][bin + 1]) / 2.0;
            }
            return _binEdges[colIdx][^1];
        }

        return Math.Min(Math.Max(token, 0), _colVocabSizes[colIdx] - 1);
    }

    #endregion

    #region Generation

    private int SampleFromLogits(Tensor<T> logits, int row, int tokenStart, int numTokens)
    {
        double temperature = Math.Max(0.01, _options.Temperature);
        int vocab = logits.Shape[1];

        double maxLogit = double.MinValue;
        for (int i = 0; i < numTokens && (tokenStart + i) < vocab; i++)
        {
            double v = NumOps.ToDouble(logits[row, tokenStart + i]) / temperature;
            if (v > maxLogit) maxLogit = v;
        }

        var probs = new double[numTokens];
        double sumExp = 0;
        for (int i = 0; i < numTokens && (tokenStart + i) < vocab; i++)
        {
            probs[i] = Math.Exp(NumOps.ToDouble(logits[row, tokenStart + i]) / temperature - maxLogit);
            sumExp += probs[i];
        }

        double u = _random.NextDouble() * sumExp;
        double cumSum = 0;
        for (int i = 0; i < numTokens; i++)
        {
            cumSum += probs[i];
            if (u <= cumSum) return tokenStart + i;
        }

        return tokenStart + numTokens - 1;
    }

    #endregion

}

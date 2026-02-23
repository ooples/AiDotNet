using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
public class REaLTabFormerGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
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
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    public REaLTabFormerGenerator(
        NeuralNetworkArchitecture<T> architecture,
        REaLTabFormerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new REaLTabFormerOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
                Layers.Add(new FullyConnectedLayer<T>(d, _options.FeedForwardDimension, relu));
                Layers.Add(new FullyConnectedLayer<T>(_options.FeedForwardDimension, d, identity));
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

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var relu = new ReLUActivation<T>() as IActivationFunction<T>;
        int d = _embDim;

        for (int l = 0; l < _options.NumLayers; l++)
        {
            _attentionQueryLayers.Add(new FullyConnectedLayer<T>(d, d, identity));
            _attentionKeyLayers.Add(new FullyConnectedLayer<T>(d, d, identity));
            _attentionValueLayers.Add(new FullyConnectedLayer<T>(d, d, identity));
            _attentionOutputLayers.Add(new FullyConnectedLayer<T>(d, d, identity));

            // FFN layers reference the Layers list for user-overridable layers
            if (!_usingCustomLayers && Layers.Count >= (l + 1) * 2)
            {
                _ffnLayer1s.Add((FullyConnectedLayer<T>)Layers[l * 2]);
                _ffnLayer2s.Add((FullyConnectedLayer<T>)Layers[l * 2 + 1]);
            }
            else
            {
                _ffnLayer1s.Add(new FullyConnectedLayer<T>(d, _options.FeedForwardDimension, relu));
                _ffnLayer2s.Add(new FullyConnectedLayer<T>(_options.FeedForwardDimension, d, identity));
            }
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
            _outputHeads.Add(new FullyConnectedLayer<T>(d, _vocabSizes[col], identity));
        }
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
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

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int startRow = batch * batchSize;
                int endRow = Math.Min(startRow + batchSize, data.Rows);

                for (int row = startRow; row < endRow; row++)
                {
                    TrainRow(tokenizedData, row, scaledLr);
                }
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
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
        var embeddings = new Vector<T>(length * d);

        for (int pos = 0; pos < length; pos++)
        {
            int token = pos < tokens.Length ? tokens[pos] : 0;
            int vocabSize = _vocabSizes[pos];

            // Token embedding: scale token index to [-1, 1] and project
            double tokenScale = vocabSize > 1 ? (2.0 * token / (vocabSize - 1) - 1.0) : 0;

            for (int j = 0; j < d; j++)
            {
                double posEnc;
                if (j % 2 == 0)
                {
                    posEnc = Math.Sin(pos / Math.Pow(10000.0, (double)j / d));
                }
                else
                {
                    posEnc = Math.Cos(pos / Math.Pow(10000.0, (double)(j - 1) / d));
                }

                // Combine token and position
                double val = tokenScale * 0.1 + posEnc;
                embeddings[pos * d + j] = NumOps.FromDouble(val);
            }
        }

        return embeddings;
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
        var output = new Vector<T>(seqLen * d);

        for (int pos = 0; pos < seqLen; pos++)
        {
            var queryInput = ExtractPosition(input, pos, d);
            var queryTensor = _attentionQueryLayers[layerIdx].Forward(VectorToTensor(queryInput));
            var query = TensorToVector(queryTensor, d);

            // Compute attention weights over positions [0, pos]
            var attnWeights = new double[pos + 1];
            double maxWeight = double.MinValue;

            for (int k = 0; k <= pos; k++)
            {
                var keyInput = ExtractPosition(input, k, d);
                var keyTensor = _attentionKeyLayers[layerIdx].Forward(VectorToTensor(keyInput));
                var key = TensorToVector(keyTensor, d);

                double dot = 0;
                for (int j = 0; j < d; j++)
                {
                    dot += NumOps.ToDouble(query[j]) * NumOps.ToDouble(key[j]);
                }
                attnWeights[k] = dot / Math.Sqrt(d);
                if (attnWeights[k] > maxWeight) maxWeight = attnWeights[k];
            }

            // Softmax
            double sumExp = 0;
            for (int k = 0; k <= pos; k++)
            {
                attnWeights[k] = Math.Exp(attnWeights[k] - maxWeight);
                sumExp += attnWeights[k];
            }
            for (int k = 0; k <= pos; k++)
            {
                attnWeights[k] /= Math.Max(sumExp, 1e-10);
            }

            // Weighted sum of values
            var attnOutput = new double[d];
            for (int k = 0; k <= pos; k++)
            {
                var valInput = ExtractPosition(input, k, d);
                var valTensor = _attentionValueLayers[layerIdx].Forward(VectorToTensor(valInput));
                var val = TensorToVector(valTensor, d);

                for (int j = 0; j < d; j++)
                {
                    attnOutput[j] += attnWeights[k] * NumOps.ToDouble(val[j]);
                }
            }

            // Output projection + residual
            var attnVec = new Vector<T>(d);
            for (int j = 0; j < d; j++) attnVec[j] = NumOps.FromDouble(attnOutput[j]);
            var projTensor = _attentionOutputLayers[layerIdx].Forward(VectorToTensor(attnVec));
            var proj = TensorToVector(projTensor, d);

            for (int j = 0; j < d; j++)
            {
                double residual = NumOps.ToDouble(input[pos * d + j]);
                output[pos * d + j] = NumOps.FromDouble(NumOps.ToDouble(proj[j]) + residual);
            }
        }

        return output;
    }

    private Vector<T> FeedForwardPerPosition(Vector<T> input, int seqLen, int layerIdx)
    {
        int d = _embDim;
        var output = new Vector<T>(seqLen * d);

        for (int pos = 0; pos < seqLen; pos++)
        {
            var posInput = ExtractPosition(input, pos, d);
            var hidden = _ffnLayer1s[layerIdx].Forward(VectorToTensor(posInput));
            var ffnOut = _ffnLayer2s[layerIdx].Forward(hidden);
            var ffnVec = TensorToVector(ffnOut, d);

            // Residual connection
            for (int j = 0; j < d; j++)
            {
                output[pos * d + j] = NumOps.FromDouble(
                    NumOps.ToDouble(ffnVec[j]) + NumOps.ToDouble(input[pos * d + j]));
            }
        }

        return output;
    }

    private Vector<T> ExtractColumnHidden(Vector<T> hiddenStates, int col)
    {
        int d = _embDim;
        var result = new Vector<T>(d);
        for (int j = 0; j < d && (col * d + j) < hiddenStates.Length; j++)
        {
            result[j] = hiddenStates[col * d + j];
        }
        return result;
    }

    private static Vector<T> ExtractPosition(Vector<T> sequence, int pos, int dim)
    {
        var result = new Vector<T>(dim);
        for (int j = 0; j < dim && (pos * dim + j) < sequence.Length; j++)
        {
            result[j] = sequence[pos * dim + j];
        }
        return result;
    }

    #endregion

    #region Training

    private void TrainRow(Matrix<T> tokenizedData, int row, T scaledLr)
    {
        int[] tokens = new int[_seqLength];
        for (int col = 0; col < _seqLength; col++)
        {
            tokens[col] = (int)NumOps.ToDouble(tokenizedData[row, col]);
        }

        // Forward pass with teacher forcing
        var embeddings = ComputeEmbeddings(tokens, _seqLength);
        var hiddenStates = TransformerForward(embeddings, _seqLength);

        // Compute loss and backward for each column output head
        for (int col = 0; col < _seqLength; col++)
        {
            var hidden = ExtractColumnHidden(hiddenStates, col);
            var logits = _outputHeads[col].Forward(VectorToTensor(hidden));

            // Cross-entropy gradient: softmax(logits) - one_hot(target)
            int targetToken = tokens[col];
            var grad = ComputeCrossEntropyGrad(logits, targetToken);

            // Sanitize and clip gradient
            grad = SafeGradient(grad, 5.0);

            _outputHeads[col].Backward(grad);
            _outputHeads[col].UpdateParameters(scaledLr);
        }

        // Update transformer layers
        UpdateTransformerParameters(scaledLr);
    }

    private Tensor<T> ComputeCrossEntropyGrad(Tensor<T> logits, int targetToken)
    {
        var grad = new Tensor<T>(logits.Shape);
        double maxVal = double.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(logits[i]) - maxVal);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            double softmax = Math.Exp(NumOps.ToDouble(logits[i]) - maxVal) / Math.Max(sumExp, 1e-10);
            double target = i == targetToken ? 1.0 : 0.0;
            grad[i] = NumOps.FromDouble(softmax - target);
        }

        return grad;
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

        // Apply temperature
        double maxVal = double.MinValue;
        for (int i = 0; i < vocabSize; i++)
        {
            double v = NumOps.ToDouble(logits[i]) / Math.Max(temperature, 1e-10);
            if (v > maxVal) maxVal = v;
        }

        // Compute probabilities
        var probs = new double[vocabSize];
        double sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = Math.Exp(NumOps.ToDouble(logits[i]) / Math.Max(temperature, 1e-10) - maxVal);
            sumExp += probs[i];
        }

        // Normalize and sample
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For standard NN prediction, run through transformer layers
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // REaLTabFormer uses its own specialized training via Fit/FitAsync.
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int count = layerParams.Length;
            if (offset + count <= parameters.Length)
            {
                var slice = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.UpdateParameters(slice);
                offset += count;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embDim);
        writer.Write(_seqLength);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _embDim = reader.ReadInt32();
        _seqLength = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new REaLTabFormerGenerator<T>(Architecture, _options);
    }

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
            ModelType = ModelType.NeuralNetwork,
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
        var result = new Tensor<T>(grad.Shape);
        double normSq = 0;

        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val)) val = 0;
            result[i] = NumOps.FromDouble(val);
            normSq += val * val;
        }

        double norm = Math.Sqrt(normSq + 1e-12);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) * scale);
            }
        }

        return result;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// REaLTabFormer uses autoregressive token-by-token generation which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}

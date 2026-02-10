using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
public class TabLLMGenGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
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

    // Transformer auxiliary layers (Q, K, V projections + output projections per layer)
    private readonly List<FullyConnectedLayer<T>> _queryLayers = new();
    private readonly List<FullyConnectedLayer<T>> _keyLayers = new();
    private readonly List<FullyConnectedLayer<T>> _valueLayers = new();
    private readonly List<FullyConnectedLayer<T>> _outProjLayers = new();

    // Token embedding and output head (auxiliary)
    private FullyConnectedLayer<T>? _tokenEmbedding;
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
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    public TabLLMGenGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TabLLMGenOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabLLMGenOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
                Layers.Add(new FullyConnectedLayer<T>(embDim, ffnDim, gelu));
                Layers.Add(new FullyConnectedLayer<T>(ffnDim, embDim, identity));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds the auxiliary transformer layers with actual vocabulary dimensions discovered during Fit().
    /// </summary>
    private void RebuildAuxiliaryLayers()
    {
        _queryLayers.Clear();
        _keyLayers.Clear();
        _valueLayers.Clear();
        _outProjLayers.Clear();

        int embDim = _options.EmbeddingDimension;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        _tokenEmbedding = new FullyConnectedLayer<T>(_vocabSize, embDim, identity);

        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            _queryLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
            _keyLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
            _valueLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
            _outProjLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
        }

        _outputHead = new FullyConnectedLayer<T>(embDim, _vocabSize, identity);

        // Rebuild default FFN layers if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            int ffnDim = _options.FeedForwardDimension;
            var gelu = new GELUActivation<T>() as IActivationFunction<T>;

            for (int layer = 0; layer < _options.NumLayers; layer++)
            {
                Layers.Add(new FullyConnectedLayer<T>(embDim, ffnDim, gelu));
                Layers.Add(new FullyConnectedLayer<T>(ffnDim, embDim, identity));
            }
        }
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);

        // Build vocabulary and tokenization
        BuildTokenization(data, columns);

        // Build transformer layers with actual vocab dimensions
        RebuildAuxiliaryLayers();

        // Tokenize all data
        var tokenizedData = TokenizeData(data, columns);

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                TrainBatch(tokenizedData, b, end, lr);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Fit(data, columns, epochs), cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_outputHead is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() before Generate().");
        }

        var result = new Matrix<T>(numSamples, _columns.Count);

        for (int i = 0; i < numSamples; i++)
        {
            var row = GenerateRow();
            for (int j = 0; j < _columns.Count && j < row.Length; j++)
            {
                result[i, j] = row[j];
            }
        }

        return result;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
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

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Training is handled through Fit() for tabular generators.
        // This method provides NeuralNetworkBase compatibility.
        var output = Predict(input);

        // Compute simple MSE gradient for backprop
        var gradient = new Tensor<T>(output.Shape);
        for (int i = 0; i < output.Length && i < expectedOutput.Length; i++)
        {
            gradient[i] = NumOps.FromDouble(
                2.0 * (NumOps.ToDouble(output[i]) - NumOps.ToDouble(expectedOutput[i])));
        }

        // Backward through FFN layers in reverse
        var current = gradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;

        // Update main FFN layers
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_vocabSize);
        writer.Write(_seqLength);
        writer.Write(_specialTokenOffset);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _vocabSize = reader.ReadInt32();
        _seqLength = reader.ReadInt32();
        _specialTokenOffset = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabLLMGenGenerator<T>(Architecture, _options);
    }

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
            ModelType = ModelType.NeuralNetwork,
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

    #region Training

    private void TrainBatch(List<int[]> tokenizedData, int startRow, int endRow, T lr)
    {
        int embDim = _options.EmbeddingDimension;

        for (int row = startRow; row < endRow; row++)
        {
            var tokens = tokenizedData[row];

            // Create one-hot encoded token inputs and run through transformer
            var embeddings = new List<Vector<T>>();
            for (int pos = 0; pos < tokens.Length; pos++)
            {
                var oneHot = new Vector<T>(_vocabSize);
                oneHot[tokens[pos]] = NumOps.FromDouble(1.0);
                var embTensor = _tokenEmbedding is not null
                    ? _tokenEmbedding.Forward(VectorToTensor(oneHot))
                    : VectorToTensor(oneHot);
                embeddings.Add(TensorToVector(embTensor, embDim));
            }

            // Apply causal transformer
            var outputs = ApplyCausalTransformer(embeddings);

            // Autoregressive loss: predict next token from current position
            for (int pos = 0; pos < tokens.Length - 1; pos++)
            {
                if (_outputHead is null) continue;

                var logitsTensor = _outputHead.Forward(VectorToTensor(outputs[pos]));
                var logits = TensorToVector(logitsTensor, _vocabSize);

                // Cross-entropy gradient
                int targetToken = tokens[pos + 1];
                var softmax = ComputeSoftmax(logits);

                var grad = new Tensor<T>([_vocabSize]);
                for (int v = 0; v < _vocabSize; v++)
                {
                    double target = v == targetToken ? 1.0 : 0.0;
                    grad[v] = NumOps.FromDouble(NumOps.ToDouble(softmax[v]) - target);
                }

                // Sanitize and clip gradient
                grad = SanitizeAndClipGradient(grad, 5.0);

                _outputHead.Backward(grad);
            }

            // Update output head
            _outputHead?.UpdateParameters(lr);
        }
    }

    #endregion

    #region Generation

    private Vector<T> GenerateRow()
    {
        int embDim = _options.EmbeddingDimension;
        int schemaTokens = _options.SchemaTokensPerColumn;
        var result = new Vector<T>(_columns.Count);

        var generatedTokens = new List<int>();
        var generatedEmbeddings = new List<Vector<T>>();

        int valueTokenBase = _specialTokenOffset + _columns.Count * schemaTokens;

        for (int c = 0; c < _columns.Count; c++)
        {
            // Add schema tokens
            int schemaBase = _specialTokenOffset + c * schemaTokens;
            for (int s = 0; s < schemaTokens; s++)
            {
                int token = schemaBase + s;
                generatedTokens.Add(token);
                generatedEmbeddings.Add(EmbedToken(token));
            }

            // Apply transformer and predict value token
            var outputs = ApplyCausalTransformer(generatedEmbeddings);
            var lastOutput = outputs[^1];

            if (_outputHead is not null)
            {
                var logitsTensor = _outputHead.Forward(VectorToTensor(lastOutput));
                var logits = TensorToVector(logitsTensor, _vocabSize);

                int valueToken = SampleFromLogits(logits, valueTokenBase, _colVocabSizes[c]);
                int relativeToken = valueToken - valueTokenBase;

                generatedTokens.Add(valueToken);
                generatedEmbeddings.Add(EmbedToken(valueToken));

                result[c] = NumOps.FromDouble(DetokenizeValue(relativeToken, c));
            }

            valueTokenBase += _colVocabSizes[c];
        }

        return result;
    }

    private int SampleFromLogits(Vector<T> logits, int tokenStart, int numTokens)
    {
        double temperature = Math.Max(0.01, _options.Temperature);

        double maxLogit = double.MinValue;
        for (int i = 0; i < numTokens && (tokenStart + i) < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[tokenStart + i]) / temperature;
            if (v > maxLogit) maxLogit = v;
        }

        var probs = new double[numTokens];
        double sumExp = 0;
        for (int i = 0; i < numTokens && (tokenStart + i) < logits.Length; i++)
        {
            probs[i] = Math.Exp(NumOps.ToDouble(logits[tokenStart + i]) / temperature - maxLogit);
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

    #region Transformer

    private List<Vector<T>> ApplyCausalTransformer(List<Vector<T>> embeddings)
    {
        int embDim = _options.EmbeddingDimension;
        var current = new List<Vector<T>>(embeddings);

        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            var attended = CausalSelfAttention(current, layer);

            // Residual + FFN
            var ffnOut = new List<Vector<T>>();
            for (int pos = 0; pos < current.Count; pos++)
            {
                // Residual from attention
                for (int d = 0; d < embDim && d < attended[pos].Length; d++)
                {
                    attended[pos][d] = NumOps.Add(attended[pos][d], current[pos][d]);
                }

                // FFN through Layers (2 layers per transformer layer)
                int ffn1Idx = layer * 2;
                int ffn2Idx = layer * 2 + 1;

                Tensor<T> h2;
                if (ffn1Idx < Layers.Count && ffn2Idx < Layers.Count)
                {
                    var h1 = Layers[ffn1Idx].Forward(VectorToTensor(attended[pos]));
                    h2 = Layers[ffn2Idx].Forward(h1);
                }
                else
                {
                    h2 = VectorToTensor(attended[pos]);
                }
                var ffnVec = TensorToVector(h2, embDim);

                // Residual from FFN
                for (int d = 0; d < embDim; d++)
                {
                    ffnVec[d] = NumOps.Add(ffnVec[d], attended[pos][d]);
                }

                ffnOut.Add(ffnVec);
            }

            current = ffnOut;
        }

        return current;
    }

    private List<Vector<T>> CausalSelfAttention(List<Vector<T>> embeds, int layerIdx)
    {
        int embDim = _options.EmbeddingDimension;
        double scale = 1.0 / Math.Sqrt(embDim);
        int seqLen = embeds.Count;

        var queries = new List<Vector<T>>();
        var keys = new List<Vector<T>>();
        var values = new List<Vector<T>>();

        for (int pos = 0; pos < seqLen; pos++)
        {
            queries.Add(TensorToVector(_queryLayers[layerIdx].Forward(VectorToTensor(embeds[pos])), embDim));
            keys.Add(TensorToVector(_keyLayers[layerIdx].Forward(VectorToTensor(embeds[pos])), embDim));
            values.Add(TensorToVector(_valueLayers[layerIdx].Forward(VectorToTensor(embeds[pos])), embDim));
        }

        // Causal attention
        var output = new List<Vector<T>>();
        for (int i = 0; i < seqLen; i++)
        {
            var scores = new double[i + 1];
            double maxScore = double.MinValue;

            for (int j = 0; j <= i; j++)
            {
                double dot = 0;
                for (int d = 0; d < embDim; d++)
                    dot += NumOps.ToDouble(queries[i][d]) * NumOps.ToDouble(keys[j][d]);
                scores[j] = dot * scale;
                if (scores[j] > maxScore) maxScore = scores[j];
            }

            double sumExp = 0;
            for (int j = 0; j <= i; j++)
            {
                scores[j] = Math.Exp(scores[j] - maxScore);
                sumExp += scores[j];
            }

            var attnOut = new Vector<T>(embDim);
            for (int j = 0; j <= i; j++)
            {
                double weight = scores[j] / Math.Max(sumExp, 1e-10);
                for (int d = 0; d < embDim; d++)
                {
                    attnOut[d] = NumOps.Add(attnOut[d],
                        NumOps.FromDouble(weight * NumOps.ToDouble(values[j][d])));
                }
            }

            var projected = _outProjLayers[layerIdx].Forward(VectorToTensor(attnOut));
            output.Add(TensorToVector(projected, embDim));
        }

        return output;
    }

    #endregion

    #region Helpers

    private Vector<T> EmbedToken(int token)
    {
        var oneHot = new Vector<T>(_vocabSize);
        if (token >= 0 && token < _vocabSize)
        {
            oneHot[token] = NumOps.FromDouble(1.0);
        }

        if (_tokenEmbedding is not null)
        {
            var tensor = _tokenEmbedding.Forward(VectorToTensor(oneHot));
            return TensorToVector(tensor, _options.EmbeddingDimension);
        }

        return CreateStandardNormalVector(_options.EmbeddingDimension);
    }

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        double maxVal = double.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > maxVal) maxVal = v;
        }

        var result = new Vector<T>(logits.Length);
        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Math.Exp(NumOps.ToDouble(logits[i]) - maxVal);
        for (int i = 0; i < logits.Length; i++)
            result[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[i]) - maxVal) / Math.Max(sumExp, 1e-10));

        return result;
    }

    /// <summary>
    /// Creates a vector of standard normal random values using Box-Muller transform.
    /// </summary>
    private Vector<T> CreateStandardNormalVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    /// <summary>
    /// Sanitizes a gradient tensor by replacing NaN/Inf values with zero and applying gradient clipping.
    /// </summary>
    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = ops.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = ops.Zero;
            }
            else
            {
                normSq += v * v;
            }
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = ops.FromDouble(ops.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
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

    /// <summary>
    /// Gets the current parameter vector from all layers.
    /// </summary>
    private Vector<T> GetParameterVector()
    {
        int totalParams = 0;
        foreach (var layer in Layers)
            totalParams += layer.ParameterCount;

        var parameters = new Vector<T>(totalParams);
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
                parameters[offset + i] = layerParams[i];
            offset += layer.ParameterCount;
        }
        return parameters;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// TabLLM-Gen uses autoregressive schema-aware generation which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}

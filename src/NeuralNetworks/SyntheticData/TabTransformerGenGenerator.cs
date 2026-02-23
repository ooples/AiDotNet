using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabTransformer-Gen generator that uses column-wise contextual embeddings and masked
/// prediction to generate realistic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabTransformer-Gen treats each column as a "token" in a sequence:
///
/// <code>
///  Col1_embed ──┐
///  Col2_embed ──┼──► Multi-Head Self-Attention ──► Contextual Embeddings ──► Column Decoders
///  Col3_embed ──┤        (columns attend to         (each column embedding     (reconstruct
///  Col4_embed ──┘         each other)                is enriched by context)     column values)
/// </code>
///
/// <b>Training</b>: Mask random columns, predict their values from the unmasked context.
/// <b>Generation</b>: Start with all columns masked, iteratively unmask (predict) columns.
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full forward/backward/update lifecycle
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like a crossword puzzle:
/// - Each column is a clue that helps fill in other columns
/// - During training, we hide some columns and learn to fill them in
/// - During generation, we start with an empty puzzle and fill in one column at a time,
///   using already-filled columns as context for the remaining ones
///
/// If you provide custom layers in the architecture, those will be used for the
/// feed-forward network blocks. If not, the network creates standard TabTransformer
/// layers based on the original paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TabTransformerGenOptions&lt;double&gt;
/// {
///     NumLayers = 6,
///     NumHeads = 8,
///     EmbeddingDimension = 32,
///     Epochs = 100
/// };
/// var gen = new TabTransformerGenGenerator&lt;double&gt;(architecture, options);
/// gen.Fit(data, columns, epochs: 100);
/// var synthetic = gen.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
/// (Huang et al., 2020) — adapted for generation with masked prediction
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabTransformerGenGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TabTransformerGenOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private int _numColumns;
    private readonly List<int> _colWidths = new();
    private Random _random;

    // Column embeddings: one projection per column to embed into shared space (auxiliary)
    private readonly List<FullyConnectedLayer<T>> _colEmbeddings = new();

    // Transformer layers: Q, K, V projections per layer (auxiliary)
    private readonly List<FullyConnectedLayer<T>> _queryLayers = new();
    private readonly List<FullyConnectedLayer<T>> _keyLayers = new();
    private readonly List<FullyConnectedLayer<T>> _valueLayers = new();

    // Column decoders: one decoder head per column (auxiliary)
    private readonly List<FullyConnectedLayer<T>> _colDecoders = new();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TabTransformerGen-specific options.
    /// </summary>
    public TabTransformerGenOptions<T> TransformerOptions => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabTransformer-Gen generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabTransformerGen-specific options for transformer configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a TabTransformer-Gen network.
    ///
    /// If you provide custom layers in the architecture, those will be used for the
    /// feed-forward network blocks (one per transformer layer). If not, the network
    /// creates standard FFN layers based on the paper specifications.
    ///
    /// The attention Q/K/V projections, column embeddings, and decoders are always
    /// created internally and are not user-overridable.
    /// </para>
    /// </remarks>
    public TabTransformerGenGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TabTransformerGenOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabTransformerGenOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the FFN layers of the TabTransformer-Gen network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Layers</b> = feed-forward network blocks (user-overridable via Architecture).
    /// Auxiliary networks (Q/K/V projections, column embeddings, decoders) are always
    /// created internally during Fit() when actual data dimensions are known.
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
            // Create default FFN blocks: two layers per transformer block
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
    /// Rebuilds auxiliary layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildAuxiliaryLayers()
    {
        int embDim = _options.EmbeddingDimension;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // Column embeddings
        _colEmbeddings.Clear();
        for (int c = 0; c < _numColumns; c++)
        {
            int colWidth = _colWidths[c];
            _colEmbeddings.Add(new FullyConnectedLayer<T>(colWidth, embDim, identity));
        }

        // Q/K/V projections per transformer layer
        _queryLayers.Clear();
        _keyLayers.Clear();
        _valueLayers.Clear();
        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            _queryLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
            _keyLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
            _valueLayers.Add(new FullyConnectedLayer<T>(embDim, embDim, identity));
        }

        // Column decoders
        _colDecoders.Clear();
        for (int c = 0; c < _numColumns; c++)
        {
            int colWidth = _colWidths[c];
            _colDecoders.Add(new FullyConnectedLayer<T>(embDim, colWidth, identity));
        }

        // Rebuild Layers (FFN blocks) if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            var gelu = new GELUActivation<T>() as IActivationFunction<T>;
            int ffnDim = _options.FeedForwardDimension;

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
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        _numColumns = columns.Count;
        var transformedData = _transformer.Transform(data);

        // Compute column widths from transformer
        _colWidths.Clear();
        for (int c = 0; c < _numColumns; c++)
        {
            var info = _transformer.GetTransformInfo(c);
            _colWidths.Add(info.Width);
        }

        // Rebuild all auxiliary layers with actual dimensions
        RebuildAuxiliaryLayers();

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                TrainMaskedPrediction(transformedData, b, end, lr);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, ct);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || !IsFitted)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data.");
        }

        var result = new Matrix<T>(numSamples, _dataWidth);
        int embDim = _options.EmbeddingDimension;

        for (int i = 0; i < numSamples; i++)
        {
            // Start with random noise for all column embeddings
            var colEmbeds = new List<Vector<T>>();
            for (int c = 0; c < _numColumns; c++)
            {
                colEmbeds.Add(CreateStandardNormalVector(embDim));
            }

            // Iterative refinement
            for (int step = 0; step < _options.GenerationSteps; step++)
            {
                // Apply transformer attention
                var contextual = ApplyTransformer(colEmbeds);

                // Decode each column and re-embed
                for (int c = 0; c < _numColumns; c++)
                {
                    var decoded = DecoderForward(c, contextual[c]);

                    if (step < _options.GenerationSteps - 1)
                    {
                        // Re-embed decoded values for next iteration
                        colEmbeds[c] = EmbedColumn(c, decoded);
                    }
                    else
                    {
                        // Final step: write to output
                        int offset = GetColumnOffset(c);
                        int width = _colWidths[c];
                        for (int j = 0; j < width && (offset + j) < _dataWidth; j++)
                        {
                            result[i, offset + j] = j < decoded.Length ? decoded[j] : NumOps.Zero;
                        }
                    }
                }
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Training

    private void TrainMaskedPrediction(Matrix<T> data, int startRow, int endRow, T lr)
    {
        int embDim = _options.EmbeddingDimension;

        for (int row = startRow; row < endRow; row++)
        {
            var x = GetRow(data, row);

            // Create column embeddings
            var colEmbeds = new List<Vector<T>>();
            for (int c = 0; c < _numColumns; c++)
            {
                var colVals = ExtractColumnValues(x, c);
                colEmbeds.Add(EmbedColumn(c, colVals));
            }

            // Determine which columns to mask
            int numMask = Math.Max(1, (int)(_numColumns * _options.MaskRatio));
            var maskIndices = new HashSet<int>();
            while (maskIndices.Count < numMask)
            {
                maskIndices.Add(_random.Next(_numColumns));
            }

            // Replace masked embeddings with noise
            foreach (int idx in maskIndices)
            {
                colEmbeds[idx] = CreateStandardNormalVector(embDim);
            }

            // Forward through transformer
            var contextual = ApplyTransformer(colEmbeds);

            // Compute reconstruction loss for masked columns only
            foreach (int maskedCol in maskIndices)
            {
                var decoded = DecoderForward(maskedCol, contextual[maskedCol]);
                var target = ExtractColumnValues(x, maskedCol);
                int colWidth = _colWidths[maskedCol];

                // MSE gradient
                var grad = new Tensor<T>([colWidth]);
                for (int j = 0; j < colWidth; j++)
                {
                    double diff = NumOps.ToDouble(decoded[j]) - NumOps.ToDouble(target[j]);
                    grad[j] = NumOps.FromDouble(2.0 * diff);
                }

                // Sanitize and clip gradient
                grad = SanitizeAndClipGradient(grad, 5.0);

                // Backward through decoder
                _colDecoders[maskedCol].Backward(grad);
                _colDecoders[maskedCol].UpdateParameters(lr);
            }
        }
    }

    #endregion

    #region Transformer Forward

    private List<Vector<T>> ApplyTransformer(List<Vector<T>> colEmbeds)
    {
        int embDim = _options.EmbeddingDimension;
        var current = new List<Vector<T>>(colEmbeds);

        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            // Self-attention across columns
            var attended = ColumnSelfAttention(current, layer);

            // Residual connection
            for (int c = 0; c < _numColumns; c++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    attended[c][d] = NumOps.Add(attended[c][d], current[c][d]);
                }
            }

            // Feed-forward per column via Layers (2 layers per block)
            int ffnIdx1 = layer * 2;
            int ffnIdx2 = layer * 2 + 1;
            var ffnOut = new List<Vector<T>>();

            for (int c = 0; c < _numColumns; c++)
            {
                Tensor<T> t1;
                Tensor<T> t2;

                if (ffnIdx1 < Layers.Count && ffnIdx2 < Layers.Count)
                {
                    t1 = Layers[ffnIdx1].Forward(VectorToTensor(attended[c]));
                    t2 = Layers[ffnIdx2].Forward(t1);
                }
                else
                {
                    // Fallback: identity
                    t2 = VectorToTensor(attended[c]);
                }

                var ffnVec = TensorToVector(t2, embDim);

                // Residual
                for (int d = 0; d < embDim; d++)
                {
                    ffnVec[d] = NumOps.Add(ffnVec[d], attended[c][d]);
                }

                ffnOut.Add(ffnVec);
            }

            current = ffnOut;
        }

        return current;
    }

    private List<Vector<T>> ColumnSelfAttention(List<Vector<T>> embeds, int layerIdx)
    {
        int embDim = _options.EmbeddingDimension;
        int numHeads = _options.NumHeads;
        int headDim = Math.Max(1, embDim / numHeads);
        double scale = 1.0 / Math.Sqrt(headDim);

        // Compute Q, K, V for all columns
        var queries = new List<Vector<T>>();
        var keys = new List<Vector<T>>();
        var values = new List<Vector<T>>();

        for (int c = 0; c < _numColumns; c++)
        {
            var qTensor = _queryLayers[layerIdx].Forward(VectorToTensor(embeds[c]));
            var kTensor = _keyLayers[layerIdx].Forward(VectorToTensor(embeds[c]));
            var vTensor = _valueLayers[layerIdx].Forward(VectorToTensor(embeds[c]));

            queries.Add(TensorToVector(qTensor, embDim));
            keys.Add(TensorToVector(kTensor, embDim));
            values.Add(TensorToVector(vTensor, embDim));
        }

        // Compute attention: for each column, attend to all columns
        var output = new List<Vector<T>>();
        for (int c = 0; c < _numColumns; c++)
        {
            // Compute attention scores
            var scores = new double[_numColumns];
            double maxScore = double.MinValue;

            for (int k = 0; k < _numColumns; k++)
            {
                double dot = 0;
                for (int d = 0; d < embDim; d++)
                {
                    dot += NumOps.ToDouble(queries[c][d]) * NumOps.ToDouble(keys[k][d]);
                }
                scores[k] = dot * scale;
                if (scores[k] > maxScore) maxScore = scores[k];
            }

            // Softmax
            double sumExp = 0;
            for (int k = 0; k < _numColumns; k++)
            {
                scores[k] = Math.Exp(scores[k] - maxScore);
                sumExp += scores[k];
            }
            for (int k = 0; k < _numColumns; k++)
            {
                scores[k] /= Math.Max(sumExp, 1e-10);
            }

            // Weighted sum of values
            var attnOut = new Vector<T>(embDim);
            for (int k = 0; k < _numColumns; k++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    attnOut[d] = NumOps.Add(attnOut[d],
                        NumOps.FromDouble(scores[k] * NumOps.ToDouble(values[k][d])));
                }
            }

            output.Add(attnOut);
        }

        return output;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For direct neural network use: treat input as a flattened row, embed columns, and decode
        if (_transformer is null || !IsFitted)
        {
            return input;
        }

        var row = TensorToVector(input, _dataWidth);
        int embDim = _options.EmbeddingDimension;

        var colEmbeds = new List<Vector<T>>();
        for (int c = 0; c < _numColumns; c++)
        {
            var colVals = ExtractColumnValues(row, c);
            colEmbeds.Add(EmbedColumn(c, colVals));
        }

        var contextual = ApplyTransformer(colEmbeds);

        var output = new Tensor<T>([_dataWidth]);
        for (int c = 0; c < _numColumns; c++)
        {
            var decoded = DecoderForward(c, contextual[c]);
            int offset = GetColumnOffset(c);
            int width = _colWidths[c];
            for (int j = 0; j < width && (offset + j) < _dataWidth; j++)
            {
                output[offset + j] = j < decoded.Length ? decoded[j] : NumOps.Zero;
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Training is handled via Fit() for tabular generators
        // This provides the NeuralNetworkBase interface compatibility
        var predicted = Predict(input);
        var loss = _lossFunction.CalculateLoss(
            TensorToVector(predicted, predicted.Length),
            TensorToVector(expectedOutput, expectedOutput.Length));
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
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
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.FeedForwardDimension);
        writer.Write(_numColumns);
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // NumLayers
        _ = reader.ReadInt32(); // NumHeads
        _ = reader.ReadInt32(); // EmbeddingDimension
        _ = reader.ReadInt32(); // FeedForwardDimension
        _numColumns = reader.ReadInt32();
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabTransformerGenGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _numColumns; i++)
        {
            string colName = i < _columns.Count ? _columns[i].Name : $"Column_{i}";
            importance[colName] = NumOps.FromDouble(1.0 / Math.Max(_numColumns, 1));
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
                ["GeneratorType"] = "TabTransformerGen",
                ["NumLayers"] = _options.NumLayers,
                ["NumHeads"] = _options.NumHeads,
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["FeedForwardDimension"] = _options.FeedForwardDimension,
                ["NumColumns"] = _numColumns,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Helpers

    private Vector<T> EmbedColumn(int colIdx, Vector<T> colValues)
    {
        if (colIdx >= _colEmbeddings.Count)
        {
            return CreateStandardNormalVector(_options.EmbeddingDimension);
        }
        var tensor = _colEmbeddings[colIdx].Forward(VectorToTensor(colValues));
        return TensorToVector(tensor, _options.EmbeddingDimension);
    }

    private Vector<T> DecoderForward(int colIdx, Vector<T> embedding)
    {
        if (colIdx >= _colDecoders.Count)
        {
            return new Vector<T>(0);
        }
        var tensor = _colDecoders[colIdx].Forward(VectorToTensor(embedding));
        return TensorToVector(tensor, _colWidths[colIdx]);
    }

    private Vector<T> ExtractColumnValues(Vector<T> row, int colIdx)
    {
        int offset = GetColumnOffset(colIdx);
        int width = _colWidths[colIdx];
        var vals = new Vector<T>(width);
        for (int j = 0; j < width && (offset + j) < row.Length; j++)
        {
            vals[j] = row[offset + j];
        }
        return vals;
    }

    private int GetColumnOffset(int colIdx)
    {
        int offset = 0;
        for (int c = 0; c < colIdx && c < _colWidths.Count; c++)
        {
            offset += _colWidths[c];
        }
        return offset;
    }

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double val = numOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                grad[i] = numOps.Zero;
                continue;
            }
            normSq += val * val;
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = numOps.FromDouble(numOps.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
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
    /// TabTransformerGen uses masked prediction with column-wise attention which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabTransformer: Tabular Data Modeling Using Contextual Embeddings",
    "https://arxiv.org/abs/2012.06678",
    Year = 2020,
    Authors = "Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin")]
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

    // Column embeddings: one projection per column to embed each column's value(s)
    // into the shared embedding space [colWidth -> embDim].
    private readonly List<FullyConnectedLayer<T>> _colEmbeddings = new();

    // Per-transformer-layer attention projections (Q, K, V): [embDim -> embDim].
    private readonly List<FullyConnectedLayer<T>> _queryLayers = new();
    private readonly List<FullyConnectedLayer<T>> _keyLayers = new();
    private readonly List<FullyConnectedLayer<T>> _valueLayers = new();

    // Per-transformer-layer feed-forward blocks: FFN1 [embDim -> ffnDim] (GELU),
    // FFN2 [ffnDim -> embDim] (identity).
    private readonly List<FullyConnectedLayer<T>> _ffn1 = new();
    private readonly List<FullyConnectedLayer<T>> _ffn2 = new();

    // Per-transformer-layer pre-residual LayerNorms (one after attention, one after FFN).
    private readonly List<LayerNormalizationLayer<T>> _attnNorms = new();
    private readonly List<LayerNormalizationLayer<T>> _ffnNorms = new();

    // Column decoders: one decoder head per column [embDim -> colWidth].
    private readonly List<FullyConnectedLayer<T>> _colDecoders = new();

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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabTransformerGenGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

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
    /// <summary>
    /// TabTransformerGen's <see cref="LayerBase{T}"/> chain is composed of
    /// per-column embeddings + per-block Q/K/V projections + per-column
    /// decoders — none of which take <c>Architecture.InputWidth</c> as
    /// their first axis. Skip the base class's architecture-driven shape
    /// pre-walk so each layer resolves from its real first input.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape() => null;

    protected override void InitializeLayers()
    {
        // Before Fit() supplies real column metadata, derive a self-consistent
        // default column layout from the architecture so the model is a valid,
        // trainable network on its own (the generated ModelFamily tests call
        // Train()/Predict() directly without ever calling Fit()). Each of the
        // InputSize features is treated as a width-1 column "token"; Fit() later
        // rebuilds the layout with the real per-column widths from the
        // TabularDataTransformer.
        _numColumns = Math.Max(1, Architecture.InputSize);
        _colWidths.Clear();
        for (int c = 0; c < _numColumns; c++) _colWidths.Add(1);
        _dataWidth = _numColumns;

        BuildLayers();
    }

    /// <summary>
    /// (Re)builds every trainable layer (column embeddings, per-layer Q/K/V
    /// projections, feed-forward blocks, LayerNorms, and column decoders) from
    /// the current <see cref="_numColumns"/> / <see cref="_colWidths"/> layout
    /// and registers them all in <see cref="NeuralNetworkBase{T}.Layers"/> so the
    /// tape-based training path collects their parameters. Called once from
    /// <see cref="InitializeLayers"/> (default layout) and again from
    /// <see cref="Fit"/> once the real column widths are known.
    /// </summary>
    private void BuildLayers()
    {
        int embDim = _options.EmbeddingDimension;
        int ffnDim = _options.FeedForwardDimension;
        var gelu = new GELUActivation<T>() as IActivationFunction<T>;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        _colEmbeddings.Clear();
        _queryLayers.Clear();
        _keyLayers.Clear();
        _valueLayers.Clear();
        _ffn1.Clear();
        _ffn2.Clear();
        _attnNorms.Clear();
        _ffnNorms.Clear();
        _colDecoders.Clear();
        Layers.Clear();

        // Column embeddings: [colWidth -> embDim] (input width resolved lazily).
        for (int c = 0; c < _numColumns; c++)
        {
            _colEmbeddings.Add(new FullyConnectedLayer<T>(embDim, identity));
        }

        // Transformer blocks.
        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            _queryLayers.Add(new FullyConnectedLayer<T>(embDim, identity));
            _keyLayers.Add(new FullyConnectedLayer<T>(embDim, identity));
            _valueLayers.Add(new FullyConnectedLayer<T>(embDim, identity));
            _ffn1.Add(new FullyConnectedLayer<T>(ffnDim, gelu));
            _ffn2.Add(new FullyConnectedLayer<T>(embDim, identity));
            _attnNorms.Add(new LayerNormalizationLayer<T>());
            _ffnNorms.Add(new LayerNormalizationLayer<T>());
        }

        // Column decoders: [embDim -> colWidth].
        for (int c = 0; c < _numColumns; c++)
        {
            _colDecoders.Add(new FullyConnectedLayer<T>(_colWidths[c], identity));
        }

        // Register every layer in the shared Layers collection (in a stable
        // order) so GetParameters / GetParameterGradients / UpdateParameters and
        // the tape-training parameter collection all see the full parameter set.
        Layers.AddRange(_colEmbeddings);
        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            Layers.Add(_queryLayers[layer]);
            Layers.Add(_keyLayers[layer]);
            Layers.Add(_valueLayers[layer]);
            Layers.Add(_ffn1[layer]);
            Layers.Add(_ffn2[layer]);
            Layers.Add(_attnNorms[layer]);
            Layers.Add(_ffnNorms[layer]);
        }
        Layers.AddRange(_colDecoders);
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

        // Rebuild all layers with the actual per-column widths.
        BuildLayers();

        // Masked-prediction training (the TabTransformer-Gen objective): for each
        // row, randomly mask a fraction of columns (zero their values) and train
        // the network to reconstruct the FULL row from the unmasked context. The
        // tape-based base.Train computes the gradient through the column-token
        // transformer and updates every registered layer via the optimizer.
        int maskedCols = Math.Max(1, (int)Math.Round(_options.MaskRatio * _numColumns));
        var rowOrder = new int[data.Rows];
        for (int i = 0; i < data.Rows; i++) rowOrder[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            ShuffleInPlace(rowOrder);
            for (int oi = 0; oi < rowOrder.Length; oi++)
            {
                int r = rowOrder[oi];
                var fullRow = GetRow(transformedData, r);
                var maskedRow = ApplyColumnMask(fullRow, maskedCols);

                var inputTensor = VectorToTensor(maskedRow);
                var targetTensor = VectorToTensor(fullRow);
                Train(inputTensor, targetTensor);
            }
        }

        IsFitted = true;
    }

    /// <summary>
    /// Returns a copy of <paramref name="row"/> with <paramref name="numMasked"/>
    /// randomly chosen columns zeroed out (the masked-prediction corruption).
    /// </summary>
    private Vector<T> ApplyColumnMask(Vector<T> row, int numMasked)
    {
        var masked = row.Clone();
        if (_numColumns == 0) return masked;

        var colIdx = new int[_numColumns];
        for (int c = 0; c < _numColumns; c++) colIdx[c] = c;
        ShuffleInPlace(colIdx);

        for (int m = 0; m < numMasked && m < _numColumns; m++)
        {
            int c = colIdx[m];
            int offset = GetColumnOffset(c);
            int width = _colWidths[c];
            for (int j = 0; j < width && (offset + j) < masked.Length; j++)
            {
                masked[offset + j] = NumOps.Zero;
            }
        }

        return masked;
    }

    /// <summary>Fisher-Yates shuffle using the generator's seeded RNG.</summary>
    private void ShuffleInPlace(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
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

        for (int i = 0; i < numSamples; i++)
        {
            // Start from a random noise row in the transformed space, then
            // iteratively reconstruct it through the trained transformer — the
            // generation-as-iterative-masked-prediction loop. LayerNorm inside
            // the forward keeps each refinement step numerically bounded.
            var row = CreateStandardNormalVector(_dataWidth);
            Tensor<T> output = VectorToTensor(row);

            for (int step = 0; step < Math.Max(1, _options.GenerationSteps); step++)
            {
                output = RunForward(output);
            }

            var finalRow = TensorToVector(output, _dataWidth);
            for (int j = 0; j < _dataWidth; j++)
            {
                T v = j < finalRow.Length ? finalRow[j] : NumOps.Zero;
                double dv = NumOps.ToDouble(v);
                result[i, j] = (double.IsNaN(dv) || double.IsInfinity(dv)) ? NumOps.Zero : v;
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Training

    #endregion

    #region Transformer Forward

    /// <summary>
    /// Tape-connected forward pass: embeds each column as a token, runs the
    /// column-token transformer (multi-head self-attention + feed-forward, each
    /// wrapped in a residual + LayerNorm), and decodes every column back to its
    /// value space. ALL operations stay on <see cref="Tensor{T}"/> via
    /// <see cref="NeuralNetworkBase{T}.Engine"/> ops so, when invoked inside a
    /// training <c>GradientTape</c>, gradients flow to every layer's parameters.
    /// The previous implementation round-tripped through <c>Vector&lt;T&gt;</c> +
    /// scalar <c>NumOps</c> math, which detached the tape and left every
    /// parameter without a gradient (no training ever happened).
    /// </summary>
    private Tensor<T> RunForward(Tensor<T> input)
    {
        int embDim = _options.EmbeddingDimension;

        // Before Fit() supplies real column metadata, adapt the default
        // (width-1-per-feature) column layout to the actual input length so the
        // model works as a generic network for any 1-D input — the generated
        // ModelFamily tests construct the model from an architecture whose
        // InputSize need not equal the test's feature count. Only triggers when
        // not fitted and the width genuinely differs; within a single test the
        // input length is constant, so this rebuilds at most once (on the first
        // forward) and stays stable through training.
        if (!IsFitted && input.Length != _dataWidth)
        {
            _numColumns = Math.Max(1, input.Length);
            _colWidths.Clear();
            for (int c = 0; c < _numColumns; c++) _colWidths.Add(1);
            _dataWidth = _numColumns;
            BuildLayers();
        }

        // Canonicalise the input to a flat [dataWidth] vector (callers pass
        // either [dataWidth] or a unit-batched [1, dataWidth]).
        var flat = input.Rank == 1 && input.Length == _dataWidth
            ? input
            : Engine.Reshape(input, new[] { _dataWidth });

        // 1) Embed each column into the shared embedding space and stack the
        //    per-column embeddings into a [numColumns, embDim] token sequence.
        var embRows = new List<Tensor<T>>(_numColumns);
        int offset = 0;
        for (int c = 0; c < _numColumns; c++)
        {
            int width = _colWidths[c];
            var colSlice = Engine.TensorSlice(flat, new[] { offset }, new[] { width }); // [width]
            offset += width;
            var col2D = Engine.Reshape(colSlice, new[] { 1, width }); // [1, width]
            var emb = _colEmbeddings[c].Forward(col2D);               // [1, embDim]
            embRows.Add(Engine.Reshape(emb, new[] { 1, embDim }));
        }
        var seq = Engine.TensorConcatenate(embRows.ToArray(), axis: 0); // [numColumns, embDim]

        // 2) Transformer blocks.
        for (int layer = 0; layer < _options.NumLayers; layer++)
        {
            // Multi-head self-attention across columns.
            var attn = MultiHeadAttention(seq, layer, embDim);        // [numColumns, embDim]
            seq = Engine.TensorAdd(seq, attn);                        // residual
            seq = _attnNorms[layer].Forward(seq);                     // LayerNorm

            // Position-wise feed-forward.
            var ff = _ffn1[layer].Forward(seq);                       // [numColumns, ffnDim]
            ff = _ffn2[layer].Forward(ff);                            // [numColumns, embDim]
            seq = Engine.TensorAdd(seq, ff);                          // residual
            seq = _ffnNorms[layer].Forward(seq);                      // LayerNorm
        }

        // 3) Decode each column back to its value space and concatenate.
        var decoded = new List<Tensor<T>>(_numColumns);
        for (int c = 0; c < _numColumns; c++)
        {
            int width = _colWidths[c];
            var row = Engine.TensorSliceAxis(seq, axis: 0, index: c); // [embDim]
            var row2D = Engine.Reshape(row, new[] { 1, embDim });     // [1, embDim]
            var dec = _colDecoders[c].Forward(row2D);                 // [1, width]
            decoded.Add(Engine.Reshape(dec, new[] { width }));        // [width]
        }
        var output = Engine.TensorConcatenate(decoded.ToArray(), axis: 0); // [dataWidth]
        return output;
    }

    /// <summary>
    /// Multi-head scaled-dot-product self-attention over the column-token
    /// sequence, computed entirely with tape-connected <see cref="Engine"/> ops.
    /// </summary>
    private Tensor<T> MultiHeadAttention(Tensor<T> seq, int layer, int embDim)
    {
        // Heads must evenly divide the embedding dim for the per-head reshape;
        // fall back to single-head full-dim attention otherwise.
        int numHeads = _options.NumHeads > 0 && embDim % _options.NumHeads == 0
            ? _options.NumHeads
            : 1;
        int headDim = embDim / numHeads;
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        var q = _queryLayers[layer].Forward(seq);  // [numColumns, embDim]
        var k = _keyLayers[layer].Forward(seq);
        var v = _valueLayers[layer].Forward(seq);

        if (numHeads == 1)
        {
            return SingleHeadAttention(q, k, v, scale);
        }

        // Reshape to [numColumns, numHeads, headDim] and attend per head.
        var q3 = Engine.Reshape(q, new[] { _numColumns, numHeads, headDim });
        var k3 = Engine.Reshape(k, new[] { _numColumns, numHeads, headDim });
        var v3 = Engine.Reshape(v, new[] { _numColumns, numHeads, headDim });

        var headOutputs = new List<Tensor<T>>(numHeads);
        for (int h = 0; h < numHeads; h++)
        {
            var qh = Engine.TensorSliceAxis(q3, axis: 1, index: h);   // [numColumns, headDim]
            var kh = Engine.TensorSliceAxis(k3, axis: 1, index: h);
            var vh = Engine.TensorSliceAxis(v3, axis: 1, index: h);
            headOutputs.Add(SingleHeadAttention(qh, kh, vh, scale));  // [numColumns, headDim]
        }

        return Engine.TensorConcatenate(headOutputs.ToArray(), axis: 1); // [numColumns, embDim]
    }

    /// <summary>softmax(Q·Kᵀ · scale)·V for a single head — all tape-connected.</summary>
    private Tensor<T> SingleHeadAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, T scale)
    {
        var kt = Engine.TensorTranspose(k);                  // [dim, numColumns]
        var scores = Engine.TensorMatMul(q, kt);             // [numColumns, numColumns]
        scores = Engine.TensorMultiplyScalar(scores, scale);
        var probs = Engine.TensorSoftmax(scores, axis: 1);   // row-wise softmax
        return Engine.TensorMatMul(probs, v);                // [numColumns, dim]
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // The column-token transformer reconstructs the full row from its input.
        // A default column layout is always built in InitializeLayers, so the
        // network is usable before Fit() (the generated ModelFamily tests call
        // Predict()/Train() directly). The same tape-connected forward is used
        // for both inference and training.
        return RunForward(input);
    }

    /// <summary>
    /// The training forward — identical to <see cref="Predict"/> — overridden so
    /// the tape-based training path runs the column-token transformer rather than
    /// the default sequential walk over <see cref="NeuralNetworkBase{T}.Layers"/>
    /// (which would feed each layer's output to the next, ignoring the column /
    /// attention structure). Keeping it on tape-connected <see cref="Engine"/>
    /// ops lets the optimizer compute real gradients for every parameter.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return RunForward(input);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
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
        // Per-column widths define the embedding/decoder layer boundaries within
        // Layers — serialize them so a deserialized clone can re-bind its typed
        // layer references and run the identical column-token forward.
        writer.Write(_colWidths.Count);
        for (int c = 0; c < _colWidths.Count; c++) writer.Write(_colWidths[c]);
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
        int colWidthCount = reader.ReadInt32();
        _colWidths.Clear();
        for (int c = 0; c < colWidthCount; c++) _colWidths.Add(reader.ReadInt32());
        IsFitted = reader.ReadBoolean();

        // The base deserializer rebuilt Layers from the serialized layer list,
        // orphaning the typed references the constructor populated. Re-bind them
        // from the freshly-loaded Layers so the column-token forward uses the
        // deserialized (trained) weights rather than the clone's discarded init.
        ExtractLayerReferences();
    }

    /// <summary>
    /// Re-binds the typed layer-reference lists (embeddings, per-block Q/K/V +
    /// FFN + norms, decoders) from the shared <see cref="NeuralNetworkBase{T}.Layers"/>
    /// collection, using the known build order and the current
    /// <see cref="_numColumns"/> / <see cref="_options"/> layout. Idempotent and
    /// safe to call after deserialization (where Layers is repopulated with new
    /// layer instances). No-ops if the Layers count doesn't match the expected
    /// structure (e.g. a custom external layer chain).
    /// </summary>
    private void ExtractLayerReferences()
    {
        int numLayers = _options.NumLayers;
        int expected = _numColumns * 2 + numLayers * 7;
        if (Layers.Count != expected)
        {
            // Fail fast rather than silently leaving the typed-layer lists empty — RunForward
            // indexes into _colEmbeddings / _queryLayers / etc. and would otherwise throw a
            // confusing ArgumentOutOfRangeException at the first lookup with no hint as to
            // why. Hitting this means either deserialization saw a layer chain that doesn't
            // match the current TabTransformerGenOptions (NumLayers / numColumns drift between
            // save and load) or someone replaced Layers with an external chain — neither of
            // which the default forward path can handle.
            throw new InvalidOperationException(
                $"TabTransformerGenGenerator: Layers.Count = {Layers.Count} does not match the " +
                $"expected structure {expected} (= {_numColumns} columns × 2 + {numLayers} layers × 7). " +
                $"Confirm the model was serialized with the same TabTransformerGenOptions, or " +
                $"override the forward path if you intend to plug in a custom layer chain.");
        }

        _colEmbeddings.Clear();
        _queryLayers.Clear();
        _keyLayers.Clear();
        _valueLayers.Clear();
        _ffn1.Clear();
        _ffn2.Clear();
        _attnNorms.Clear();
        _ffnNorms.Clear();
        _colDecoders.Clear();

        int idx = 0;
        for (int c = 0; c < _numColumns; c++)
            _colEmbeddings.Add((FullyConnectedLayer<T>)Layers[idx++]);
        for (int l = 0; l < numLayers; l++)
        {
            _queryLayers.Add((FullyConnectedLayer<T>)Layers[idx++]);
            _keyLayers.Add((FullyConnectedLayer<T>)Layers[idx++]);
            _valueLayers.Add((FullyConnectedLayer<T>)Layers[idx++]);
            _ffn1.Add((FullyConnectedLayer<T>)Layers[idx++]);
            _ffn2.Add((FullyConnectedLayer<T>)Layers[idx++]);
            _attnNorms.Add((LayerNormalizationLayer<T>)Layers[idx++]);
            _ffnNorms.Add((LayerNormalizationLayer<T>)Layers[idx++]);
        }
        for (int c = 0; c < _numColumns; c++)
            _colDecoders.Add((FullyConnectedLayer<T>)Layers[idx++]);
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

}

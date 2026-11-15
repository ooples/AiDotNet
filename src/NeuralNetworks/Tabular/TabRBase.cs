using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Distributions;
using AiDotNet.Factories;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Attributes;
using AiDotNet.Models.Parameters;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of TabR, a retrieval-augmented model for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabR combines neural networks with instance-based learning. It encodes features,
/// retrieves similar training samples, and uses attention to aggregate neighbor
/// information for making predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR is like having a photographic memory for training data.
///
/// Architecture overview:
/// 1. **Feature Encoder**: MLP that converts raw features to embeddings
/// 2. **Retrieval Index**: Stores embeddings of all training samples
/// 3. **K-NN Search**: Finds the K most similar training samples
/// 4. **Context Encoder**: Uses attention to aggregate neighbor information
/// 5. **Prediction Head**: Makes final prediction using combined information
///
/// Why this approach works:
/// - Tabular data often has "local" structure (similar inputs → similar outputs)
/// - Neural networks alone may struggle with rare patterns
/// - Retrieval provides explicit "memory" of similar past examples
/// - Attention learns which neighbors are most relevant
///
/// Think of it as:
/// - Traditional k-NN: Just average neighbors (simple but limited)
/// - TabR: Learn to encode, then learn to attend to neighbors (powerful)
/// </para>
/// <para>
/// The architecture is the reference <c>yandex-research/tabular-dl-tabr</c> <c>bin/tabr.py</c> model. The encoder
/// is <c>x~ = Linear(d_in, d_main)(x)</c> followed by <see cref="TabROptions{T}.EncoderNumBlocks"/> residual blocks
/// <c>x~ = x~ + Dropout(dropout1)(Linear(d_block, d_main)(Dropout(dropout0)(Activation(Linear(d_main, d_block)(Norm(x~))))))</c>,
/// the first without its norm. The retrieval module follows Gorishniy et al. (2024), section 3: the target and
/// every candidate are encoded, <c>k = W_K(x~)</c> (of <c>Norm(x~)</c> under mixer normalization); the m
/// candidates with the smallest <c>||k - k_i||^2</c> are retrieved; their weights are
/// <c>softmax(-||k - k_i||^2)</c>, dropped by <c>context_dropout</c>; each neighbour contributes the value
/// <c>W_Y(y_i) + T(k - k_i)</c>, with <c>T = LinearWithoutBias(Dropout(Activation(Linear(.))))</c> and
/// <c>W_Y</c> the label embedding; and the predictor runs <see cref="TabROptions{T}.PredictorNumBlocks"/>
/// pre-normalized residual blocks over <c>x~ + sum_i w_i V_i</c>, then the head <c>Activation(Norm(.))</c> whose
/// final linear map a subclass adds. Candidates are re-encoded on every pass, so the encoder and <c>W_K</c> also
/// learn through the retrieved neighbours. The neighbour labels enter only when the index was built with labels
/// and <see cref="TabROptions{T}.IncludeNeighborTargets"/> is on.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class TabRBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly TabROptions<T> Options;

    /// <summary>
    /// Number of input features.
    /// </summary>
    protected readonly int NumFeatures;

    // Encoder E: the optional numeric embedding (num_embeddings), Linear(d_in, d_main), then the encoder_n_blocks
    // residual blocks
    private readonly PeriodicNumericEmbedding<T>? _numericEmbedding;
    private readonly FullyConnectedLayer<T> _inputProjection;
    private readonly List<TabRResidualBlock<T>> _encoderBlocks;

    // Width of every block's hidden layer and of T's: d_block = int(d_main * d_multiplier)
    private readonly int _blockDimension;

    // Every normalization layer built, switched to training mode only for a training step (BatchNorm's batch statistics)
    private readonly List<ILayer<T>> _normalizationLayers;

    // Retrieval index (stores training sample embeddings)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexEmbeddings;       // [numTrainSamples, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexFeatures;         // [numTrainSamples, numFeatures]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexLabels;           // [numTrainSamples, labelDimension] (one-hot or targets)
    private int _numIndexedSamples;

    // Label embedding W_Y of the retrieved neighbours' labels (null when labels are not used)
    private readonly FullyConnectedLayer<T>? _labelEmbedding;
    private readonly int _labelDimension;

    // Predictor P: predictor_n_blocks pre-normalized residual blocks, then the head Activation(Norm(.)) whose final
    // Linear(d_main, d_out) a subclass adds
    private readonly List<TabRResidualBlock<T>> _predictorBlocks;
    private readonly ILayer<T>? _headNormalization;
    private readonly ActivationLayer<T> _headActivation;

    // Retrieval: the optional mixer normalization before W_K, the key projection W_K, and the two linear maps of the
    // value module T(k - k_i)
    private readonly ILayer<T>? _mixerNormalization;
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // The generator of the label-encoder initialization and of the training-step dropout masks, which are drawn
    // through CompiledStepRandom
    private readonly Random _random;

    // True only while a training step runs: the reference model drops in train mode and never at prediction
    private bool _inTrainingStep;

    // Cache for backward pass
    [Scratch]
    private Tensor<T>? _attentionWeightsCache;
    [Scratch]
    private Matrix<int>? _neighborIndicesCache;
    [Scratch]
    [KeepsActivation("Read after the pass by PredictWithConfidence; holds labels, not activations.")]
    private Tensor<T>? _neighborLabelsCache;   // [batch, m, labelDimension]

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => Options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of nearest neighbors to retrieve.
    /// </summary>
    public int NumNeighbors => Options.NumNeighbors;

    /// <summary>
    /// Gets whether the retrieval index has been built.
    /// </summary>
    public bool IsIndexBuilt => _indexEmbeddings != null && _numIndexedSamples > 0;

    /// <summary>
    /// Gets the number of samples in the retrieval index.
    /// </summary>
    public int NumIndexedSamples => _numIndexedSamples;

    /// <summary>
    /// Trainable layers a subclass adds on top of the backbone, walked after it by every
    /// parameter surface below.
    /// </summary>
    /// <remarks>
    /// The task head. A classifier and a regressor share this whole backbone and differ only by a
    /// final projection, and each re-implemented the surfaces just to append that one layer. The
    /// subclass now states WHAT it adds and the folds below decide where it goes, so the count and
    /// the vector cannot place it differently. Mirrors NeuralNetworkBase.GetExtraTrainableLayers,
    /// which solves the same problem for models that derive from it -- this base has no base at
    /// all, so it cannot inherit that one.
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedLayers<T>(this, typeof(TabRBase<T>));

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// Count, read and restore all derive from THIS. Before, the count and the read were two
    /// separate walks restating the same component order, and there was no restore at all -- so the
    /// vector this model produced could be saved and never loaded back. Two walks agree until
    /// someone adds a component to one of them, and nothing reports the disagreement because the
    /// lengths still look plausible; one enumeration makes that unrepresentable.
    /// </remarks>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            var registry = new ParameterComponentRegistry<T>();

            if (_numericEmbedding is not null) registry.Register("00000000/embedding", _numericEmbedding);
            registry.Register("00000000/input", _inputProjection);

            // Optional components are registered only when built: a null entry is a shape-deferred slot,
            // which makes every parameter read and count throw ParameterLayoutNotReadyException.
            RegisterBlocks(registry, "00000001", _encoderBlocks);
            if (_mixerNormalization is not null) registry.Register("00000002/mixerNorm", _mixerNormalization);

            registry.Register("00000005/key", _keyProjection);
            registry.Register("00000006/value", _valueProjection);
            registry.Register("00000007/output", _outputProjection);
            if (_labelEmbedding is not null) registry.Register("00000008/labelEmbedding", _labelEmbedding);

            RegisterBlocks(registry, "00000009", _predictorBlocks);
            if (_headNormalization is not null) registry.Register("00000010/headNorm", _headNormalization);

            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);
            }

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the TabRBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    /// <param name="labelDimension">
    /// Width of one encoded label: the class count for a classifier (one-hot), the output dimension for a
    /// regressor. Zero means the model never embeds neighbour labels.
    /// </param>
    /// <param name="categoricalLabels">
    /// True when the labels are one-hot classes. The reference then embeds them with <c>nn.Embedding(n_classes, d_main)</c>,
    /// a bias-free map initialized uniform on [-1, 1]; continuous targets get <c>nn.Linear(1, d_main)</c>, weight and
    /// bias uniform on [-1/sqrt(2), 1/sqrt(2)].
    /// </param>
    protected TabRBase(
        int numFeatures, TabROptions<T>? options = null, int labelDimension = 0, bool categoricalLabels = false)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabROptions<T>();
        NumFeatures = numFeatures;
        if (labelDimension < 0)
            throw new ArgumentOutOfRangeException(nameof(labelDimension), "Label dimension cannot be negative.");
        _labelDimension = labelDimension;

        // Validate the configuration as the reference Model.__init__ does
        int embDim = Options.EmbeddingDimension;
        if (embDim < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TabROptions<T>.EmbeddingDimension), embDim, "The embedding dimension d_main must be positive.");
        }

        double blockWidth = embDim * Options.BlockWidthMultiplier;
        if (double.IsNaN(blockWidth) || double.IsInfinity(blockWidth) || blockWidth < 1 || blockWidth > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TabROptions<T>.BlockWidthMultiplier), Options.BlockWidthMultiplier,
                "EmbeddingDimension * BlockWidthMultiplier must give a block width d_block of at least 1.");
        }

        // d_block = int(d_main * d_multiplier), truncating as Python's int() does for a positive value
        _blockDimension = (int)blockWidth;

        if (Options.EncoderNumBlocks < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TabROptions<T>.EncoderNumBlocks), Options.EncoderNumBlocks,
                "The encoder block count cannot be negative.");
        }

        if (Options.PredictorNumBlocks < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(TabROptions<T>.PredictorNumBlocks), Options.PredictorNumBlocks,
                "The predictor block count cannot be negative.");
        }

        // mixer_normalization 'auto' resolves to encoder_n_blocks > 0, and the reference asserts that an encoder
        // without blocks has none
        bool mixerNormalization = Options.MixerNormalization ?? Options.EncoderNumBlocks > 0;
        if (mixerNormalization && Options.EncoderNumBlocks == 0)
        {
            throw new ArgumentException(
                "MixerNormalization needs at least one encoder block: the reference model asserts that an encoder with " +
                "encoder_n_blocks = 0 has no mixer normalization. Leave it null (auto) or set EncoderNumBlocks.",
                nameof(options));
        }

        ValidateDropoutRate(Options.DropoutRate, nameof(TabROptions<T>.DropoutRate));
        ValidateDropoutRate(Options.BlockOutputDropoutRate, nameof(TabROptions<T>.BlockOutputDropoutRate));
        ValidateDropoutRate(Options.ContextDropoutRate, nameof(TabROptions<T>.ContextDropoutRate));

        // The activation must be element-wise: the factory rejects Softmax and any undefined value here, at construction
        _ = CreateActivation();

        _normalizationLayers = new List<ILayer<T>>();

        // num_embeddings: d_in = n_num_features * d_embedding under an embedding, else n_num_features
        int encoderInputWidth = numFeatures;
        switch (Options.NumericEmbedding)
        {
            case TabRNumericEmbedding.None:
                break;
            case TabRNumericEmbedding.PeriodicLinearReLU:
                if (numFeatures < 1)
                {
                    throw new ArgumentException(
                        "A numeric embedding has one set of frequencies per feature, so it needs the feature count.",
                        nameof(numFeatures));
                }

                _numericEmbedding = new PeriodicNumericEmbedding<T>(
                    numFeatures, Options.NumFrequencies, Options.FrequencyInitScale, Options.NumericEmbeddingDimension,
                    Options.LiteNumericEmbedding);
                encoderInputWidth = _numericEmbedding.OutputWidth;
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(TabROptions<T>.NumericEmbedding), Options.NumericEmbedding, "Unknown TabR numeric embedding.");
        }

        // Encoder E: self.linear = nn.Linear(d_in, d_main) with no activation, then blocks0 = [make_block(i > 0)]. A
        // non-positive feature count defers the input width to the first forward.
        _inputProjection = encoderInputWidth > 0
            ? new FullyConnectedLayer<T>(encoderInputWidth, embDim, (IActivationFunction<T>?)null)
            : new FullyConnectedLayer<T>(embDim, (IActivationFunction<T>?)null);
        _encoderBlocks = new List<TabRResidualBlock<T>>();
        for (int i = 0; i < Options.EncoderNumBlocks; i++)
        {
            _encoderBlocks.Add(CreateBlock(prenorm: i > 0));
        }

        // Retrieval module R: the mixer normalization, K = nn.Linear(d_main, d_main), and
        // T = Sequential(Linear(d_main, d_block), Activation, Dropout(dropout0), Linear(d_block, d_main, bias=False))
        _mixerNormalization = mixerNormalization ? CreateNormalization() : null;
        _keyProjection = new FullyConnectedLayer<T>(embDim, embDim, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(embDim, _blockDimension, CreateActivation());
        // T's last map is bias-free in the paper and the reference; the bias stays configurable through
        // ValueOutputBiasMode
        _outputProjection = new FullyConnectedLayer<T>(
            _blockDimension, embDim, (IActivationFunction<T>?)null, Options.ValueOutputBiasMode);

        // Predictor P: blocks1 = [make_block(True)], head = Sequential(Norm, Activation, Linear(d_main, d_out)). The
        // final linear map is the subclass's task head.
        _predictorBlocks = new List<TabRResidualBlock<T>>();
        for (int i = 0; i < Options.PredictorNumBlocks; i++)
        {
            _predictorBlocks.Add(CreateBlock(prenorm: true));
        }

        _headNormalization = CreateNormalization();
        _headActivation = new ActivationLayer<T>(CreateActivation());

        // Normalization layers start in inference mode; TrainWithTape switches them on for its step only
        foreach (var normalization in _normalizationLayers)
        {
            normalization.SetTrainingMode(false);
        }

        // One generator for the numeric-embedding and label-encoder initializations and, after them, every
        // training-step dropout mask
        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _numericEmbedding?.Initialize(_random);

        if (Options.IncludeNeighborTargets && labelDimension > 0)
        {
            // Label encoder W_Y. Classes: nn.Embedding(n_classes, d_main) of the one-hot label is the bias-free map whose
            // weight column c is the embedding of class c, initialized uniform on [-1, 1]. Targets: nn.Linear(1, d_main)
            // with weight and bias uniform on [-1/sqrt(2), 1/sqrt(2)] (reference reset_parameters), widened here to
            // every output of a multi-output regressor.
            _labelEmbedding = new FullyConnectedLayer<T>(
                labelDimension, embDim, (IActivationFunction<T>?)null,
                categoricalLabels ? BiasMode.Never : BiasMode.Always);
            double bound = categoricalLabels ? 1.0 : 1.0 / Math.Sqrt(2.0);
            var initial = AiDotNet.Training.CompiledStepRandom<T>.Uniform(
                new[] { (int)_labelEmbedding.ParameterCount }, _random, -bound, bound);
            _labelEmbedding.SetParameters(initial.ToVector());
        }
    }

    /// <summary>
    /// The reference <c>make_block(prenorm)</c>: <c>[Norm(d_main)] -&gt; Linear(d_main, d_block) -&gt; Activation -&gt;
    /// Dropout(dropout0) -&gt; Linear(d_block, d_main) -&gt; Dropout(dropout1)</c>, run residually by
    /// <see cref="ForwardResidualBlock"/>.
    /// </summary>
    private TabRResidualBlock<T> CreateBlock(bool prenorm)
    {
        int embDim = Options.EmbeddingDimension;
        var normalization = prenorm ? CreateNormalization() : null;
        return new TabRResidualBlock<T>(
            normalization,
            new FullyConnectedLayer<T>(embDim, _blockDimension, CreateActivation()),
            new FullyConnectedLayer<T>(_blockDimension, embDim, (IActivationFunction<T>?)null));
    }

    /// <summary>
    /// The reference <c>Normalization(d_main)</c> for <see cref="TabROptions{T}.Normalization"/>, or null for
    /// <see cref="TabRNormalization.Identity"/>.
    /// </summary>
    private ILayer<T>? CreateNormalization()
    {
        int embDim = Options.EmbeddingDimension;
        ILayer<T>? layer = Options.Normalization switch
        {
            TabRNormalization.LayerNorm => new LayerNormalizationLayer<T>(embDim),
            TabRNormalization.BatchNorm => new BatchNormalizationLayer<T>(embDim),
            TabRNormalization.Identity => null,
            _ => throw new ArgumentOutOfRangeException(
                nameof(TabROptions<T>.Normalization), Options.Normalization, "Unknown TabR normalization.")
        };

        if (layer is not null) _normalizationLayers.Add(layer);
        return layer;
    }

    /// <summary>A fresh instance of the reference <c>Activation()</c> for <see cref="TabROptions{T}.Activation"/>.</summary>
    private IActivationFunction<T> CreateActivation()
        => ActivationFunctionFactory<T>.CreateActivationFunction(Options.Activation);

    private static void RegisterBlocks(
        ParameterComponentRegistry<T> registry, string prefix, List<TabRResidualBlock<T>> blocks)
    {
        for (int i = 0; i < blocks.Count; i++)
        {
            var block = blocks[i];
            if (block.Normalization is not null) registry.Register($"{prefix}/{i:D8}/00000000", block.Normalization);
            registry.Register($"{prefix}/{i:D8}/00000001", block.Expand);
            registry.Register($"{prefix}/{i:D8}/00000002", block.Project);
        }
    }

    /// <summary>
    /// One residual block, <c>x + Dropout(dropout1)(Project(Dropout(dropout0)(Expand(Norm(x)))))</c>: the reference's
    /// <c>x = x + block(x)</c>. Both dropouts are training-step masks, so outside a step the block is deterministic.
    /// </summary>
    internal Tensor<T> ForwardResidualBlock(TabRResidualBlock<T> block, Tensor<T> x)
    {
        if (block is null) throw new ArgumentNullException(nameof(block));
        if (x is null) throw new ArgumentNullException(nameof(x));

        var hidden = block.Normalization is null ? x : block.Normalization.Forward(x);
        hidden = TrainingDropout(block.Expand.Forward(hidden), Options.DropoutRate);
        hidden = TrainingDropout(block.Project.Forward(hidden), Options.BlockOutputDropoutRate);
        return Engine.TensorAdd(x, hidden);
    }

    /// <summary>The encoder's first linear map, <c>Linear(d_in, d_main)</c>.</summary>
    internal FullyConnectedLayer<T> InputProjection => _inputProjection;

    /// <summary>The numeric-feature embedding before the encoder, or null without one.</summary>
    internal PeriodicNumericEmbedding<T>? NumericEmbeddingModule => _numericEmbedding;

    /// <summary>The encoder's residual blocks, the first without its normalization.</summary>
    internal IReadOnlyList<TabRResidualBlock<T>> EncoderBlocks => _encoderBlocks;

    /// <summary>The normalization before the key projection, or null when mixer normalization is off.</summary>
    internal ILayer<T>? MixerNormalizationLayer => _mixerNormalization;

    /// <summary>The key projection <c>K = Linear(d_main, d_main)</c>.</summary>
    internal FullyConnectedLayer<T> KeyProjection => _keyProjection;

    /// <summary>The first map of <c>T</c>, <c>Activation(Linear(d_main, d_block))</c>.</summary>
    internal FullyConnectedLayer<T> ValueHiddenProjection => _valueProjection;

    /// <summary>The last map of <c>T</c>, <c>Linear(d_block, d_main)</c>, bias-free by default.</summary>
    internal FullyConnectedLayer<T> ValueOutputProjection => _outputProjection;

    /// <summary>The label encoder <c>W_Y</c>, or null when neighbour labels are not used.</summary>
    internal FullyConnectedLayer<T>? LabelEmbedding => _labelEmbedding;

    /// <summary>The predictor's pre-normalized residual blocks.</summary>
    internal IReadOnlyList<TabRResidualBlock<T>> PredictorBlocks => _predictorBlocks;

    /// <summary>The head's normalization, or null for <see cref="TabRNormalization.Identity"/>.</summary>
    internal ILayer<T>? HeadNormalization => _headNormalization;

    /// <summary>The hidden width <c>d_block</c> of every block and of <c>T</c>.</summary>
    internal int BlockDimension => _blockDimension;

    /// <summary>Every parameterized backbone layer, in the order of the parameter registry.</summary>
    private IEnumerable<ILayer<T>> BackboneLayers()
    {
        yield return _inputProjection;
        foreach (var block in _encoderBlocks)
        {
            foreach (var layer in block.Layers) yield return layer;
        }

        if (_mixerNormalization is not null) yield return _mixerNormalization;
        yield return _keyProjection;
        yield return _valueProjection;
        yield return _outputProjection;
        if (_labelEmbedding is not null) yield return _labelEmbedding;
        foreach (var block in _predictorBlocks)
        {
            foreach (var layer in block.Layers) yield return layer;
        }

        if (_headNormalization is not null) yield return _headNormalization;
    }

    /// <summary>
    /// Encodes input features to embeddings: the reference encoder <c>x = Linear(num_embeddings(x)); x = x + block(x)</c>
    /// for every encoder block.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Embeddings [batch_size, embedding_dim].</returns>
    protected Tensor<T> EncodeFeatures(Tensor<T> features)
    {
        var x = _inputProjection.Forward(_numericEmbedding is null ? features : _numericEmbedding.Forward(features));
        foreach (var block in _encoderBlocks)
        {
            x = ForwardResidualBlock(block, x);
        }

        // Optionally normalize to unit length for retrieval
        if (Options.NormalizeEmbeddings)
        {
            x = NormalizeEmbeddings(x);
        }

        return x;
    }

    /// <summary>
    /// Normalizes embeddings to unit length.
    /// </summary>
    private Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings)
    {
        int batchSize = embeddings.Shape[0];
        int dim = embeddings.Shape[1];
        var epsilon = NumOps.FromDouble(1e-12);

        // L2 norm of every row (with epsilon inside the square root), then divide the row by it
        var rows = embeddings.Reshape(batchSize, dim);
        var normSq = Engine.ReduceSum(Engine.TensorMultiply(rows, rows), new[] { 1 }, true);
        var norm = Engine.TensorSqrt(TensorTapeOps.TapeAddScalar(Engine, normSq, epsilon));
        var normalized = Engine.TensorDivide(rows, norm).Reshape(embeddings._shape);

        return normalized;
    }

    /// <summary>
    /// Builds the retrieval index from training data.
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates the "memory" of training samples.
    /// Call this once after training or when you want to update the index.
    ///
    /// The index stores:
    /// - Encoded embeddings of all training samples
    /// - Original features (for reference)
    ///
    /// During prediction, we search this index to find similar samples.
    /// </para>
    /// </remarks>
    public void BuildIndex(Tensor<T> features)
    {
        _numIndexedSamples = features.Shape[0];
        // A copy, as the labels are: every forward re-encodes the indexed features, so a reference would let the
        // caller's later writes to its own tensor change the index
        _indexFeatures = TabularMetrics.KeptCopy(features);
        _indexLabels = null;

        // Encode all training samples; the index outlives this call, so it holds its own copy of the engine result
        _indexEmbeddings = TabularMetrics.KeptCopy(EncodeFeatures(features));
    }

    /// <summary>
    /// Builds the retrieval index from training data and the encoded labels of every candidate, which the
    /// label embedding <c>W_Y</c> turns into part of each neighbour's value.
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <param name="labelEncodings">Encoded labels [num_samples, labelDimension].</param>
    protected void BuildIndexWithLabels(Tensor<T> features, Tensor<T> labelEncodings)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (labelEncodings is null) throw new ArgumentNullException(nameof(labelEncodings));
        int samples = features.Shape[0];
        if (_labelDimension == 0 || labelEncodings.Length != samples * _labelDimension)
        {
            throw new ArgumentException(
                $"Labels must hold {_labelDimension} values for each of the {samples} indexed samples.",
                nameof(labelEncodings));
        }

        BuildIndex(features);
        _indexLabels = TabularMetrics.KeptCopy(labelEncodings).Reshape(samples, _labelDimension);
    }

    /// <summary>
    /// Retrieves the K nearest indexed samples for query samples, by squared L2 distance in embedding space.
    /// </summary>
    /// <param name="queryEmbeddings">Query embeddings [batch_size, embedding_dim].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for leave-one-out during training).</param>
    /// <returns>Tuple of (neighbor indices matrix, neighbor embeddings, squared distances).</returns>
    protected (Matrix<int> Indices, Tensor<T> Embeddings, Tensor<T> Distances) RetrieveNeighbors(
        Tensor<T> queryEmbeddings,
        Vector<int>? excludeIndices = null)
    {
        if (_indexEmbeddings == null)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int embDim = queryEmbeddings.Shape[1];
        var (nearest, neighbors, distances) = RetrieveNearest(
            queryEmbeddings.Reshape(queryEmbeddings.Shape[0], embDim),
            _indexEmbeddings.Reshape(_numIndexedSamples, embDim),
            excludeIndices);
        return (nearest.ToMatrix(), neighbors, distances);
    }

    /// <summary>
    /// The m candidates nearest to each query by squared L2 distance <c>||q - c_i||^2</c>, in stable ascending
    /// order of distance, with each query's excluded candidate (leave-one-out) never retrieved.
    /// </summary>
    /// <returns>The neighbour indices [batch, m], the neighbour rows [batch, m, dim] and their distances.</returns>
    private (Tensor<int> Nearest, Tensor<T> Neighbors, Tensor<T> Distances) RetrieveNearest(
        Tensor<T> queries,
        Tensor<T> candidates,
        Vector<int>? excludeIndices)
    {
        int batchSize = queries.Shape[0];
        int dim = queries.Shape[1];

        // The target is never its own candidate (paper section 3.1). An excluded column is pushed to the end of
        // the ranking, so it must also fall outside the k retrieved, or it would be weighted like a neighbour.
        bool excludes = excludeIndices != null && excludeIndices.Length > 0;
        int k = Math.Min(Options.NumNeighbors, _numIndexedSamples - (excludes ? 1 : 0));
        if (k < 1)
        {
            throw new InvalidOperationException(
                "Retrieval needs at least one candidate besides the excluded sample; index more samples.");
        }

        // The ranking is a discrete selection with no gradient, so it is computed off the tape. Distances are direct
        // per-row differences: the ||q||^2 + ||c||^2 - 2 q.c expansion cancels for keys that are close relative to
        // their norm, misranks the neighbours in float and overflows the squares in Half.
        Tensor<int> nearest;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var allDistances = TabularMetrics.SquaredDistances(queries, candidates);

            // Skip excluded indices (for leave-one-out): row b excludes column excludeIndices[b] when b is covered
            if (excludes && excludeIndices != null)
            {
                int covered = Math.Min(batchSize, excludeIndices.Length);
                var excluded = Tensor<int>.FromVector(excludeIndices.GetSubVector(0, covered));
                if (covered < batchSize)
                {
                    excluded = Engine.TensorConcatenate(
                        new[] { excluded, Engine.TensorFullLike(new Tensor<int>(new[] { batchSize - covered }), -1) }, 0);
                }

                var excludedMask = Engine.TensorOneHot<T>(excluded, _numIndexedSamples).Reshape(batchSize, _numIndexedSamples);
                allDistances = Engine.TensorWhere(
                    excludedMask, Engine.TensorFullLike(allDistances, NumOps.MaxValue), allDistances);
            }

            // k nearest in stable ascending order of distance
            nearest = TieBreakArgMax.StableSmallestK(allDistances, k);
        }

        // The gathered rows and their distances are on the tape, so gradients reach the queries and candidates
        var neighbors = Engine.TensorIndexSelect(candidates, nearest.Reshape(batchSize * k), 0)
            .Reshape(batchSize, k, dim);
        var offsets = Engine.TensorSubtract(queries.Reshape(batchSize, 1, dim), neighbors);
        var distances = Engine.ReduceSum(Engine.TensorMultiply(offsets, offsets), new[] { 2 }, false);

        return (nearest, neighbors, distances);
    }

    /// <summary>
    /// The retrieval module of Gorishniy et al. (2024), eq. 5: weights <c>softmax(-||k - k_i||^2 / temperature)</c>
    /// over the retrieved neighbours, values <c>W_Y(y_i) + T(k - k_i)</c>, and their weighted sum.
    /// </summary>
    /// <param name="queryKeys">Target keys k [batch_size, embedding_dim].</param>
    /// <param name="neighborKeys">Retrieved keys k_i [batch_size, m, embedding_dim].</param>
    /// <param name="neighborLabels">Encoded neighbour labels [batch_size * m, labelDimension], or null.</param>
    /// <returns>Aggregated context [batch_size, embedding_dim].</returns>
    private Tensor<T> AggregateNeighbors(Tensor<T> queryKeys, Tensor<T> neighborKeys, Tensor<T>? neighborLabels)
    {
        int batchSize = queryKeys.Shape[0];
        int embDim = queryKeys.Shape[1];
        int k = neighborKeys.Shape[1];

        // k - k_i for every neighbour, on the tape
        var difference = Engine.TensorSubtract(queryKeys.Reshape(batchSize, 1, embDim), neighborKeys);

        // S = -||k - k_i||^2, divided by the retrieval temperature (1 in the paper)
        var squaredDistance = Engine.ReduceSum(Engine.TensorMultiply(difference, difference), new[] { 2 }, false);
        var scores = TensorTapeOps.TapeMultiplyScalar(
            Engine, squaredDistance, NumOps.FromDouble(-1.0 / Options.RetrievalTemperature));
        var attentionWeights = Engine.Softmax(scores, -1);
        _attentionWeightsCache = TabularMetrics.KeptCopy(attentionWeights);

        // The reference applies context_dropout to the retrieval weights (probs = self.dropout(probs))
        attentionWeights = TrainingDropout(attentionWeights, Options.ContextDropoutRate);

        // V_i = T(k - k_i) + W_Y(y_i), with T = LinearWithoutBias(Dropout(Activation(Linear(.)))); the activation is
        // part of the first map
        var differenceRows = difference.Reshape(batchSize * k, embDim);
        var hidden = TrainingDropout(_valueProjection.Forward(differenceRows), Options.DropoutRate);
        var values = _outputProjection.Forward(hidden);
        if (_labelEmbedding != null && neighborLabels != null)
        {
            values = Engine.TensorAdd(values, _labelEmbedding.Forward(neighborLabels));
        }

        // sum_i w_i V_i
        return Engine.ReduceSum(
            Engine.TensorMultiply(attentionWeights.Reshape(batchSize, k, 1), values.Reshape(batchSize, k, embDim)),
            new[] { 1 },
            false);
    }

    /// <summary>
    /// Performs the forward pass through the TabR backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for training).</param>
    /// <returns>Combined representation [batch_size, embedding_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features, Vector<int>? excludeIndices = null)
    {
        if (_indexFeatures == null || _numIndexedSamples == 0)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int batchSize = features.Shape[0];
        int embDim = Options.EmbeddingDimension;

        // Encode the targets and, with the same encoder, every candidate; keys k = W_K(x~), or W_K(Norm(x~)) under
        // mixer normalization
        var queryEmbedding = EncodeFeatures(features).Reshape(batchSize, embDim);
        var candidateEmbedding = EncodeFeatures(_indexFeatures).Reshape(_numIndexedSamples, embDim);
        // A training step encodes the candidates with dropout, so only an inference encoding refreshes the index
        if (!_inTrainingStep)
        {
            _indexEmbeddings = TabularMetrics.KeptCopy(candidateEmbedding);
        }
        var queryKeys = ComputeKeys(queryEmbedding);
        var candidateKeys = ComputeKeys(candidateEmbedding);

        // Retrieve the m nearest candidates by key distance
        var (nearest, neighborKeys, _) = RetrieveNearest(queryKeys, candidateKeys, excludeIndices);
        _neighborIndicesCache = nearest.ToMatrix();

        int k = nearest.Shape[1];
        Tensor<T>? neighborLabels = null;
        _neighborLabelsCache = null;
        if (_indexLabels != null)
        {
            neighborLabels = Engine.TensorIndexSelect(_indexLabels, nearest.Reshape(batchSize * k), 0);
            _neighborLabelsCache = TabularMetrics.KeptCopy(neighborLabels).Reshape(batchSize, k, _labelDimension);
        }

        // x = x~ + sum_i w_i V_i, then the predictor blocks x = x + block(x)
        var context = Engine.TensorAdd(queryEmbedding, AggregateNeighbors(queryKeys, neighborKeys, neighborLabels));
        foreach (var block in _predictorBlocks)
        {
            context = ForwardResidualBlock(block, context);
        }

        // The head's Norm and Activation; the subclass applies its final Linear(d_main, d_out)
        if (_headNormalization != null)
        {
            context = _headNormalization.Forward(context);
        }

        return _headActivation.Forward(context);
    }

    /// <summary>The reference key <c>k = K(x if normalization is None else normalization(x))</c>.</summary>
    private Tensor<T> ComputeKeys(Tensor<T> embeddings)
        => _keyProjection.Forward(_mixerNormalization is null ? embeddings : _mixerNormalization.Forward(embeddings));

    /// <summary>
    /// The live tensors a tape training step updates: the encoder, the retrieval projections, the label
    /// embedding, the predictor and the task head a subclass adds.
    /// </summary>
    protected IEnumerable<Tensor<T>> GetTrainableTensors()
    {
        var layers = new List<ILayer<T>>(BackboneLayers());
        layers.AddRange(GetExtraTrainableLayers());
        var tensors = new List<Tensor<T>>(TabularTraining.LayerTensors(layers));
        if (_numericEmbedding is not null) tensors.AddRange(_numericEmbedding.Tensors);
        return tensors;
    }

    /// <summary>
    /// One tape training step: gradients of <paramref name="objective"/> over every trainable tensor, then SGD.
    /// </summary>
    /// <returns>The loss before the update.</returns>
    protected T TrainWithTape(Func<Tensor<T>> objective, T learningRate)
    {
        // Dropout and batch statistics are active for the step only; prediction never drops
        _inTrainingStep = true;
        SetNormalizationTrainingMode(true);
        try
        {
            return TabularTraining.SgdStep(GetTrainableTensors, objective, learningRate);
        }
        finally
        {
            _inTrainingStep = false;
            SetNormalizationTrainingMode(false);
            ResetState();
        }
    }

    private void SetNormalizationTrainingMode(bool isTraining)
    {
        foreach (var normalization in _normalizationLayers)
        {
            normalization.SetTrainingMode(isTraining);
        }
    }

    /// <summary>
    /// The reference TabR's dropout (<c>nn.Dropout</c> in train mode): inside a training step each element is kept with
    /// probability 1 - <paramref name="rate"/> and scaled by 1 / (1 - <paramref name="rate"/>); otherwise the input.
    /// </summary>
    /// <remarks>
    /// The uniform draws come from <see cref="AiDotNet.Training.CompiledStepRandom{T}.Uniform"/>, so a compiled training plan records them
    /// and refills them before every replay instead of freezing the first step's mask into the plan. The mask is a
    /// constant of the step, and the gradient flows through the kept elements.
    /// </remarks>
    private Tensor<T> TrainingDropout(Tensor<T> input, double rate)
    {
        if (!_inTrainingStep || rate <= 0)
        {
            return input;
        }

        var draws = AiDotNet.Training.CompiledStepRandom<T>.Uniform(input.Shape.ToArray(), _random);
        var keep = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(draws, NumOps.FromDouble(rate)));
        return Engine.TensorMultiply(input, Engine.TensorMultiplyScalar(keep, NumOps.FromDouble(1.0 / (1.0 - rate))));
    }

    private static void ValidateDropoutRate(double rate, string name)
    {
        if (double.IsNaN(rate) || rate < 0 || rate >= 1)
        {
            throw new ArgumentOutOfRangeException(name, rate, "A dropout rate must lie in [0, 1).");
        }
    }

    /// <summary>
    /// The candidates a training batch must not retrieve: <paramref name="sampleIndices"/> when given, otherwise each
    /// row's own entry in the index.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper (section 3.1) and the reference code exclude every training object from its own candidate set.
    /// Values carry the label embedding <c>W_Y(y_i)</c>, so a row that retrieves itself (at distance 0, with the
    /// largest weight) is trained on its own label. Without indices, a row is identified with the indexed sample whose
    /// features it matches exactly; a row matching none is not in the index and excludes nothing.
    /// </para>
    /// <para>
    /// A row matching several indexed samples exactly (duplicate rows) cannot be told apart from its twins, so its own
    /// entry is unknown; that throws rather than silently leaking a label. Pass <paramref name="sampleIndices"/> then.
    /// </para>
    /// </remarks>
    /// <returns>The per-row excluded index (-1 for none), or null when no row needs one.</returns>
    protected Vector<int>? ResolveTrainingExclusions(Tensor<T> features, Vector<int>? sampleIndices)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (sampleIndices != null) return sampleIndices;
        if (_indexFeatures == null || _numIndexedSamples == 0)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int batchSize = features.Shape[0];
        int numFeatures = _indexFeatures.Shape[1];
        if (batchSize == 0) return null;

        int probe = Math.Min(2, _numIndexedSamples);
        Tensor<int> nearest;
        Tensor<T> exact;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var rows = TabularMetrics.Flat(features, batchSize * numFeatures).Reshape(batchSize, numFeatures);
            var distances = TabularMetrics.SquaredDistances(rows, _indexFeatures);
            nearest = TieBreakArgMax.StableSmallestK(distances, probe);
            exact = Engine.TensorEquals(Engine.TensorTakeAlongDim(distances, nearest, 1), NumOps.Zero);
        }

        if (probe > 1 && Engine.TensorCountNonzero(Engine.TensorNarrow(exact, 1, 1, 1)) > 0)
        {
            throw new InvalidOperationException(
                "A training row matches several indexed samples exactly, so its own index entry cannot be identified " +
                "and it would retrieve its own label. Pass sampleIndices with each row's index in BuildIndex.");
        }

        var matched = Engine.TensorNarrow(exact, 1, 0, 1).Reshape(batchSize);
        if (Engine.TensorCountNonzero(matched) == 0) return null;

        var self = Engine.TensorNarrow(nearest, 1, 0, 1).Reshape(batchSize);
        var none = Engine.TensorFullLike(self, -1);
        return Engine.TensorWhere(matched.Cast<int>(), self, none).ToVector();
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // The encoder, the retrieval module (mixer norm, K, T, W_Y) and the predictor
        foreach (var layer in BackboneLayers())
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    /// <remarks>
    /// This model had no restore path at all. Its vector could be produced and saved, and there was
    /// no way to load it back -- so a checkpoint of a TabR model was write-only.
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
        => ParameterRegistry.SetParameters(parameters);

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _attentionWeightsCache = null;
        _neighborIndicesCache = null;
        _neighborLabelsCache = null;

        foreach (var layer in BackboneLayers())
        {
            layer.ResetState();
        }

        _headActivation.ResetState();
    }

    /// <summary>
    /// Gets the attention weights from the last forward pass.
    /// </summary>
    /// <returns>Attention weights [batch_size, num_neighbors] or null if not available.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention weights show how much the model focuses on each neighbor:
    /// - Higher weights = neighbor is more influential in the prediction
    /// - Useful for interpretability (understanding why a prediction was made)
    /// - Can identify which training examples are most relevant
    /// </para>
    /// </remarks>
    public Tensor<T>? GetAttentionWeights() => _attentionWeightsCache;

    /// <summary>
    /// Gets the retrieved neighbor indices from the last forward pass.
    /// </summary>
    /// <returns>Neighbor indices matrix [batch_size, num_neighbors] or null if not available.</returns>
    public Matrix<int>? GetRetrievedNeighborIndices() => _neighborIndicesCache;

    /// <summary>
    /// The encoded labels [batch, m, labelDimension] of the neighbours the last forward pass retrieved, or null
    /// when the index holds no labels.
    /// </summary>
    protected Tensor<T>? GetNeighborLabels() => _neighborLabelsCache;
}

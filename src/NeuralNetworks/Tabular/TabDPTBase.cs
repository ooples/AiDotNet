using AiDotNet.ActivationFunctions;
using System;
using System.Collections.Generic;
using AiDotNet.Models.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for TabDPT (Tabular Data Pre-Training) foundation model.
/// </summary>
/// <remarks>
/// <para>
/// TabDPT applies foundation model concepts to tabular data, using pre-training
/// on diverse datasets to learn transferable representations that can adapt
/// to new tasks through in-context learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDPT is like a "GPT for tables":
///
/// - **Pre-training**: Model learns patterns from many different tabular datasets
/// - **Transfer learning**: These learned patterns help on new, unseen data
/// - **In-context learning**: Given a few examples, it adapts to new tasks
/// - **Feature-wise attention**: Understands relationships between columns
///
/// The model processes features as tokens and uses transformer architecture
/// to capture complex interactions, similar to how language models process words.
/// </para>
/// <para>
/// Reference: "TabDPT: Scaling Tabular Foundation Models" (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TabDPTBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected readonly TabDPTOptions<T> Options;
    protected readonly Random _random;

    // Feature embedding
    private readonly FullyConnectedLayer<T> _featureProjection;
    private readonly FullyConnectedLayer<T>[] _categoricalEmbeddings;

    // Transformer layers
    private readonly TransformerBlock<T>[] _transformerBlocks;

    // Optional feature-wise attention
    private readonly FeatureAttentionBlock<T>? _featureAttention;

    // MLP head for final representation
    private readonly FullyConnectedLayer<T>[] _mlpLayers;
    private readonly LayerNormalizationLayer<T> _finalNorm;

    // Cached values
    private Tensor<T>? _embeddingsCache;
    private Tensor<T>? _transformerOutputCache;
    private Tensor<T>? _mlpOutputCache;

    /// <summary>
    /// Gets the number of numerical features.
    /// </summary>
    public int NumNumericalFeatures { get; }

    /// <summary>
    /// Gets the MLP output dimension.
    /// </summary>
    protected int MLPOutputDimension => Options.OutputHeadDimensions[^1];

    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// Extra trainable layers a subclass contributes, folded after the shared backbone.
    /// </summary>
    /// <remarks>
    /// The regression and classification variants share this whole backbone and differ only by a
    /// final projection. Each used to override <see cref="ParameterCount"/> purely to append that
    /// one layer -- and because this base had no GetParameters or SetParameters at all, the head was
    /// COUNTED and never read, never restored and never checkpointed. The count grew; the model that
    /// could be saved did not. Declaring the head here means the subclass states WHAT it adds and
    /// the registry decides where it goes, so count, vector and restore cannot disagree about it.
    /// </remarks>
    protected virtual IEnumerable<IParameterSource<T>> GetExtraTrainableLayers()
        => System.Linq.Enumerable.Empty<IParameterSource<T>>();

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Count, read and restore all derive from THIS, rather than each restating the component list.
    /// Three parallel walks are how a count and a vector come to describe different models: they
    /// agree until someone adds a component to two of them, and nothing reports the disagreement
    /// because the lengths still look plausible. One enumeration makes that unrepresentable.
    /// </para>
    /// <para>
    /// The stable IDs carry a numeric prefix because the registry orders by identity rather than by
    /// the order Register happened to be called -- so the prefix, not the call order, is what pins
    /// serialization order, and it survives a component being added in the middle later.
    /// </para>
    /// </remarks>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            var registry = new ParameterComponentRegistry<T>();
            registry.Register("0000/projection", _featureProjection);

            for (int i = 0; i < _categoricalEmbeddings.Length; i++)
                registry.Register($"0001/{i:D8}", _categoricalEmbeddings[i]);

            for (int i = 0; i < _transformerBlocks.Length; i++)
                registry.Register($"0002/{i:D8}", _transformerBlocks[i]);

            registry.Register("0003/featureAttention", _featureAttention);

            for (int i = 0; i < _mlpLayers.Length; i++)
                registry.Register($"0004/{i:D8}", _mlpLayers[i]);

            registry.Register("0005/finalNorm", _finalNorm);
            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) registry.Register($"9000/{extraIndex++:D8}", extra);
            }

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <summary>
    /// Reads every parameter in traversal order. <see cref="SetParameters"/> reads it back in the
    /// same order, and <see cref="ParameterCount"/> is the length of what this returns.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

    /// <summary>
    /// Initializes a new instance of the TabDPTBase class.
    /// </summary>
    protected TabDPTBase(int numNumericalFeatures, TabDPTOptions<T>? options = null)
    {
        if (numNumericalFeatures < 0)
        {
            throw new ArgumentException("Number of features cannot be negative", nameof(numNumericalFeatures));
        }

        Options = options ?? new TabDPTOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        _random = RandomHelper.CreateSecureRandom();

        int embDim = Options.EmbeddingDimension;

        // Feature projection for numerical features
        _featureProjection = numNumericalFeatures > 0
            ? new FullyConnectedLayer<T>(
                numNumericalFeatures,
                embDim,
                Options.InputActivation ?? new ReLUActivation<T>())
            : new FullyConnectedLayer<T>(
                embDim,
                Options.InputActivation ?? new ReLUActivation<T>());

        // Categorical embeddings
        var cardinalities = Options.CategoricalCardinalities ?? [];
        _categoricalEmbeddings = new FullyConnectedLayer<T>[cardinalities.Length];

        for (int i = 0; i < cardinalities.Length; i++)
        {
            _categoricalEmbeddings[i] = new FullyConnectedLayer<T>(
                cardinalities[i],
                embDim,
                (IActivationFunction<T>?)null);
        }

        // Transformer blocks
        _transformerBlocks = new TransformerBlock<T>[Options.NumLayers];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _transformerBlocks[i] = new TransformerBlock<T>(
                embDim,
                Options.NumHeads,
                Options.FeedForwardDimension,
                Options.DropoutRate,
                Options.UsePreNorm,
                Options.InitScale,
                _random);
        }

        // Optional feature-wise attention
        if (Options.UseFeatureAttention)
        {
            _featureAttention = new FeatureAttentionBlock<T>(
                embDim,
                Options.NumHeads,
                Options.DropoutRate,
                _random);
        }

        // MLP head
        var mlpDims = Options.OutputHeadDimensions;
        _mlpLayers = new FullyConnectedLayer<T>[mlpDims.Length];

        int inputDim = embDim;
        for (int i = 0; i < mlpDims.Length; i++)
        {
            bool isLast = i == mlpDims.Length - 1;
            _mlpLayers[i] = new FullyConnectedLayer<T>(
                inputDim,
                mlpDims[i],
                isLast ? null : Options.HiddenActivation ?? new GELUActivation<T>());
            inputDim = mlpDims[i];
        }

        _finalNorm = new LayerNormalizationLayer<T>(inputDim);
    }

    /// <summary>
    /// Performs the forward pass through the backbone network.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        int embDim = Options.EmbeddingDimension;

        // Project numerical features
        var embeddings = _featureProjection.Forward(numericalFeatures);
        _embeddingsCache = embeddings;

        // Add categorical embeddings if present
        if (categoricalIndices != null && _categoricalEmbeddings.Length > 0)
        {
            var cardinalities = Options.CategoricalCardinalities
                ?? throw new InvalidOperationException("TabDPTBase: CategoricalCardinalities not configured.");

            for (int catIdx = 0; catIdx < _categoricalEmbeddings.Length; catIdx++)
            {
                var oneHot = CreateOneHotEncoding(
                    categoricalIndices,
                    catIdx,
                    cardinalities[catIdx]);
                var catEmb = _categoricalEmbeddings[catIdx].Forward(oneHot);

                // Add to embeddings
                embeddings = Engine.TensorAdd(embeddings, catEmb);
            }
        }

        // Apply transformer blocks
        var transformerOutput = embeddings;
        foreach (var block in _transformerBlocks)
        {
            transformerOutput = block.Forward(transformerOutput);
        }
        _transformerOutputCache = transformerOutput;

        // Apply feature-wise attention if enabled
        if (_featureAttention != null)
        {
            transformerOutput = _featureAttention.Forward(transformerOutput);
        }

        // Apply MLP head
        var mlpOutput = transformerOutput;
        foreach (var layer in _mlpLayers)
        {
            mlpOutput = layer.Forward(mlpOutput);
        }

        // Final normalization
        mlpOutput = _finalNorm.Forward(mlpOutput);
        _mlpOutputCache = mlpOutput;

        return mlpOutput;
    }

    /// <summary>
    /// Creates one-hot encoding for categorical features.
    /// </summary>
    private Tensor<T> CreateOneHotEncoding(Matrix<int> categoricalIndices, int featureIndex, int cardinality)
    {
        int batchSize = categoricalIndices.Rows;
        var oneHot = new Tensor<T>([batchSize, cardinality]);

        for (int b = 0; b < batchSize; b++)
        {
            int categoryIndex = categoricalIndices[b, featureIndex];
            if (categoryIndex >= 0 && categoryIndex < cardinality)
            {
                oneHot[b * cardinality + categoryIndex] = NumOps.One;
            }
        }

        return oneHot;
    }

    /// <summary>
    /// Updates all trainable parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        _featureProjection.UpdateParameters(learningRate);

        foreach (var emb in _categoricalEmbeddings)
        {
            emb.UpdateParameters(learningRate);
        }

        foreach (var block in _transformerBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        _featureAttention?.UpdateParameters(learningRate);

        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        _finalNorm.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state and caches.
    /// </summary>
    public virtual void ResetState()
    {
        _embeddingsCache = null;
        _transformerOutputCache = null;
        _mlpOutputCache = null;

        _featureProjection.ResetState();

        foreach (var emb in _categoricalEmbeddings)
        {
            emb.ResetState();
        }

        foreach (var block in _transformerBlocks)
        {
            block.ResetState();
        }

        _featureAttention?.ResetState();

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }

        _finalNorm.ResetState();
    }

    /// <summary>
    /// Transformer block with multi-head attention and feed-forward network.
    /// </summary>
    private sealed class TransformerBlock<TBlock> : IParameterSource<TBlock>
    {
        private static readonly INumericOperations<TBlock> NumOps = MathHelper.GetNumericOperations<TBlock>();

        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly bool _usePreNorm;
        private readonly double _dropoutRate;

        // Attention weights
        private Tensor<TBlock> _queryWeights;
        private Tensor<TBlock> _keyWeights;
        private Tensor<TBlock> _valueWeights;
        private Tensor<TBlock> _outputWeights;

        // Attention gradients
        private Tensor<TBlock> _queryGrad;
        private Tensor<TBlock> _keyGrad;
        private Tensor<TBlock> _valueGrad;
        private Tensor<TBlock> _outputGrad;

        // Feed-forward layers
        private readonly FullyConnectedLayer<TBlock> _ff1;
        private readonly FullyConnectedLayer<TBlock> _ff2;

        // Layer norms
        private readonly LayerNormalizationLayer<TBlock> _norm1;
        private readonly LayerNormalizationLayer<TBlock> _norm2;

        // Cached values
        private Tensor<TBlock>? _inputCache;
        private Tensor<TBlock>? _normInput1Cache;
        private Tensor<TBlock>? _attentionOutputCache;
        private Tensor<TBlock>? _queryCache;
        private Tensor<TBlock>? _keyCache;
        private Tensor<TBlock>? _valueCache;
        private Tensor<TBlock>? _attentionScoresCache;

        /// <summary>The block attention projections, in serialization order.</summary>
        private IEnumerable<Tensor<TBlock>> AttentionTensors()
        {
            yield return _queryWeights;
            yield return _keyWeights;
            yield return _valueWeights;
            yield return _outputWeights;
        }

        /// <summary>The block sub-layers, after the attention projections.</summary>
        private IEnumerable<ILayer<TBlock>> SubLayers()
        {
            yield return _ff1;
            yield return _ff2;
            yield return _norm1;
            yield return _norm2;
        }

        /// <summary>
        /// Summed from the SAME traversal the vector uses, so the two cannot disagree.
        /// </summary>
        /// <remarks>
        /// This was the formula <c>_embeddingDim * _embeddingDim * 4</c> plus the sub-layers. A
        /// formula restates what the tensors already know and drifts from them silently -- and with
        /// no read path there was no vector to contradict it, so any error was unobservable rather
        /// than absent. The identical formula in TabPFN block proved to be wrong by 7,873.
        /// </remarks>
        public long ParameterCount
        {
            get
            {
                long count = 0;
                foreach (var tensor in AttentionTensors()) count += tensor.Length;
                foreach (var layer in SubLayers()) count += layer.ParameterCount;
                return count;
            }
        }

        /// <inheritdoc />
        public Vector<TBlock> GetParameters()
        {
            var result = new Vector<TBlock>(checked((int)ParameterCount));
            int offset = 0;

            foreach (var tensor in AttentionTensors())
            {
                for (int i = 0; i < tensor.Length; i++) result[offset++] = tensor[i];
            }

            foreach (var layer in SubLayers())
            {
                var part = layer.GetParameters();
                for (int i = 0; i < part.Length; i++) result[offset++] = part[i];
            }

            return result;
        }

        /// <summary>Writes THROUGH the attention tensors, then down into each sub-layer.</summary>
        public void SetParameters(Vector<TBlock> parameters)
        {
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            long expected = ParameterCount;
            if (parameters.Length != expected)
            {
                throw new ArgumentException(
                    $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
            }

            int offset = 0;
            foreach (var tensor in AttentionTensors())
            {
                for (int i = 0; i < tensor.Length; i++) tensor[i] = parameters[offset++];
            }

            foreach (var layer in SubLayers())
            {
                int count = checked((int)layer.ParameterCount);
                if (count == 0) continue;

                var slice = new Vector<TBlock>(count);
                for (int i = 0; i < count; i++) slice[i] = parameters[offset++];
                layer.SetParameters(slice);
            }
        }

        public TransformerBlock(
            int embeddingDim,
            int numHeads,
            int ffDim,
            double dropoutRate,
            bool usePreNorm,
            double initScale,
            Random random)
        {
            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _headDim = embeddingDim / numHeads;
            _usePreNorm = usePreNorm;
            _dropoutRate = dropoutRate;

            // Initialize attention weights
            _queryWeights = InitializeWeights([embeddingDim, embeddingDim], initScale, random);
            _keyWeights = InitializeWeights([embeddingDim, embeddingDim], initScale, random);
            _valueWeights = InitializeWeights([embeddingDim, embeddingDim], initScale, random);
            _outputWeights = InitializeWeights([embeddingDim, embeddingDim], initScale, random);

            // Initialize gradients
            _queryGrad = new Tensor<TBlock>([embeddingDim, embeddingDim]);
            _keyGrad = new Tensor<TBlock>([embeddingDim, embeddingDim]);
            _valueGrad = new Tensor<TBlock>([embeddingDim, embeddingDim]);
            _outputGrad = new Tensor<TBlock>([embeddingDim, embeddingDim]);

            // Feed-forward network
            _ff1 = new FullyConnectedLayer<TBlock>(
                embeddingDim,
                ffDim,
                new GELUActivation<TBlock>() as IActivationFunction<TBlock>);

            _ff2 = new FullyConnectedLayer<TBlock>(
                ffDim,
                embeddingDim,
                (IActivationFunction<TBlock>?)null);

            // Layer normalizations
            _norm1 = new LayerNormalizationLayer<TBlock>(embeddingDim);
            _norm2 = new LayerNormalizationLayer<TBlock>(embeddingDim);
        }

        private static Tensor<TBlock> InitializeWeights(int[] shape, double scale, Random random)
        {
            var weights = new Tensor<TBlock>(shape);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.FromDouble(random.NextGaussian() * scale);
            }
            return weights;
        }

        public Tensor<TBlock> Forward(Tensor<TBlock> input)
        {
            _inputCache = input;
            int batchSize = input.Shape[0];

            Tensor<TBlock> x;
            if (_usePreNorm)
            {
                x = _norm1.Forward(input);
                _normInput1Cache = x;
            }
            else
            {
                x = input;
            }

            // Multi-head self-attention
            var query = MatMul(x, _queryWeights);
            var key = MatMul(x, _keyWeights);
            var value = MatMul(x, _valueWeights);

            _queryCache = query;
            _keyCache = key;
            _valueCache = value;

            var attentionOutput = ComputeAttention(query, key, value, batchSize);
            attentionOutput = MatMul(attentionOutput, _outputWeights);
            _attentionOutputCache = attentionOutput;

            // Residual connection
            var eng = AiDotNetEngine.Current;
            var residual1 = eng.TensorAdd(input, attentionOutput);

            if (!_usePreNorm)
            {
                residual1 = _norm1.Forward(residual1);
            }

            // Feed-forward with pre-norm
            Tensor<TBlock> ffInput;
            if (_usePreNorm)
            {
                ffInput = _norm2.Forward(residual1);
            }
            else
            {
                ffInput = residual1;
            }

            var ffOutput = _ff1.Forward(ffInput);
            ffOutput = _ff2.Forward(ffOutput);

            // Residual connection
            var output = eng.TensorAdd(residual1, ffOutput);

            if (!_usePreNorm)
            {
                output = _norm2.Forward(output);
            }

            return output;
        }

        private Tensor<TBlock> ComputeAttention(Tensor<TBlock> query, Tensor<TBlock> key, Tensor<TBlock> value, int batchSize)
        {
            var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

            // Compute attention scores: Q * K^T / sqrt(d_k)
            var scores = new Tensor<TBlock>([batchSize, batchSize]);

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    var dot = NumOps.Zero;
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(
                            query[i * _embeddingDim + k],
                            key[j * _embeddingDim + k]));
                    }
                    scores[i * batchSize + j] = NumOps.Multiply(dot, scale);
                }
            }

            // Apply softmax
            for (int i = 0; i < batchSize; i++)
            {
                var maxVal = scores[i * batchSize];
                for (int j = 1; j < batchSize; j++)
                {
                    var val = scores[i * batchSize + j];
                    if (NumOps.Compare(val, maxVal) > 0)
                        maxVal = val;
                }

                var sumExp = NumOps.Zero;
                for (int j = 0; j < batchSize; j++)
                {
                    var expVal = NumOps.Exp(NumOps.Subtract(scores[i * batchSize + j], maxVal));
                    scores[i * batchSize + j] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }

                for (int j = 0; j < batchSize; j++)
                {
                    scores[i * batchSize + j] = NumOps.Divide(scores[i * batchSize + j], sumExp);
                }
            }

            _attentionScoresCache = scores;

            // Compute output: softmax(scores) * V
            var output = new Tensor<TBlock>([batchSize, _embeddingDim]);

            for (int i = 0; i < batchSize; i++)
            {
                for (int k = 0; k < _embeddingDim; k++)
                {
                    var sum = NumOps.Zero;
                    for (int j = 0; j < batchSize; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            scores[i * batchSize + j],
                            value[j * _embeddingDim + k]));
                    }
                    output[i * _embeddingDim + k] = sum;
                }
            }

            return output;
        }

        private static Tensor<TBlock> MatMul(Tensor<TBlock> input, Tensor<TBlock> weights)
        {
            return AiDotNetEngine.Current.TensorMatMul(input, weights);
        }

        public void UpdateParameters(TBlock learningRate)
        {
            // Update attention weights
            var eng = AiDotNetEngine.Current;
            _queryWeights = eng.TensorSubtract(_queryWeights, eng.TensorMultiplyScalar(_queryGrad, learningRate));
            _keyWeights = eng.TensorSubtract(_keyWeights, eng.TensorMultiplyScalar(_keyGrad, learningRate));
            _valueWeights = eng.TensorSubtract(_valueWeights, eng.TensorMultiplyScalar(_valueGrad, learningRate));
            _outputWeights = eng.TensorSubtract(_outputWeights, eng.TensorMultiplyScalar(_outputGrad, learningRate));

            _ff1.UpdateParameters(learningRate);
            _ff2.UpdateParameters(learningRate);
            _norm1.UpdateParameters(learningRate);
            _norm2.UpdateParameters(learningRate);
        }

        public void ResetState()
        {
            _inputCache = null;
            _normInput1Cache = null;
            _attentionOutputCache = null;
            _queryCache = null;
            _keyCache = null;
            _valueCache = null;
            _attentionScoresCache = null;

            // Zero gradients
            var eng = AiDotNetEngine.Current;
            eng.TensorFill(_queryGrad, NumOps.Zero);
            eng.TensorFill(_keyGrad, NumOps.Zero);
            eng.TensorFill(_valueGrad, NumOps.Zero);
            eng.TensorFill(_outputGrad, NumOps.Zero);

            _ff1.ResetState();
            _ff2.ResetState();
            _norm1.ResetState();
            _norm2.ResetState();
        }
    }

    /// <summary>
    /// Feature-wise attention block for column interactions.
    /// </summary>
    private sealed class FeatureAttentionBlock<TBlock> : IParameterSource<TBlock>
    {
        private static readonly INumericOperations<TBlock> NumOps = MathHelper.GetNumericOperations<TBlock>();

        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly double _dropoutRate;

        private Tensor<TBlock> _featureQuery;
        private Tensor<TBlock> _featureKey;
        private Tensor<TBlock> _featureValue;
        private Tensor<TBlock> _featureOutput;

        private Tensor<TBlock>? _inputCache;

        /// <summary>The four projections, in serialization order.</summary>
        private IEnumerable<Tensor<TBlock>> Tensors()
        {
            yield return _featureQuery;
            yield return _featureKey;
            yield return _featureValue;
            yield return _featureOutput;
        }

        /// <inheritdoc />
        public long ParameterCount
        {
            get
            {
                long count = 0;
                foreach (var tensor in Tensors()) count += tensor.Length;
                return count;
            }
        }

        /// <inheritdoc />
        public Vector<TBlock> GetParameters()
        {
            var result = new Vector<TBlock>(checked((int)ParameterCount));
            int offset = 0;

            foreach (var tensor in Tensors())
            {
                for (int i = 0; i < tensor.Length; i++) result[offset++] = tensor[i];
            }

            return result;
        }

        /// <inheritdoc />
        public void SetParameters(Vector<TBlock> parameters)
        {
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            long expected = ParameterCount;
            if (parameters.Length != expected)
            {
                throw new ArgumentException(
                    $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
            }

            int offset = 0;
            foreach (var tensor in Tensors())
            {
                for (int i = 0; i < tensor.Length; i++) tensor[i] = parameters[offset++];
            }
        }

        public FeatureAttentionBlock(int embeddingDim, int numHeads, double dropoutRate, Random random)
        {
            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _dropoutRate = dropoutRate;

            double scale = 0.02;
            _featureQuery = InitializeWeights([embeddingDim, embeddingDim], scale, random);
            _featureKey = InitializeWeights([embeddingDim, embeddingDim], scale, random);
            _featureValue = InitializeWeights([embeddingDim, embeddingDim], scale, random);
            _featureOutput = InitializeWeights([embeddingDim, embeddingDim], scale, random);
        }

        private static Tensor<TBlock> InitializeWeights(int[] shape, double scale, Random random)
        {
            var weights = new Tensor<TBlock>(shape);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.FromDouble(random.NextGaussian() * scale);
            }
            return weights;
        }

        public Tensor<TBlock> Forward(Tensor<TBlock> input)
        {
            _inputCache = input;
            // For feature attention, we transpose and apply attention across features
            // This is a simplified implementation
            return input;
        }

        public Tensor<TBlock> Backward(Tensor<TBlock> gradient)
        {
            return gradient;
        }

        public void UpdateParameters(TBlock learningRate)
        {
            // Parameters updated during backward pass
        }

        public void ResetState()
        {
            _inputCache = null;
        }
    }
}

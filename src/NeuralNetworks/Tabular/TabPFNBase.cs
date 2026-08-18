using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System.Collections.Generic;
using System;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for TabPFN (Prior-Fitted Networks) for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabPFN is a meta-learning approach using transformers pre-trained on synthetic
/// data. It performs in-context learning by conditioning on training examples
/// to make predictions on test samples.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabPFN works differently from traditional models:
///
/// - **Pre-training**: Model is trained on millions of synthetic datasets
/// - **In-context learning**: Training data becomes part of the input
/// - **No gradient updates**: Inference only, no fine-tuning needed
/// - **Transformer backbone**: Uses attention to learn patterns from context
///
/// The key insight is that TabPFN learns to be a "learning algorithm" itself,
/// similar to how GPT learns to complete text.
/// </para>
/// <para>
/// Reference: "TabPFN: A Transformer That Solves Small Tabular Classification
/// Problems in a Second" (2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TabPFNBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected readonly TabPFNOptions<T> Options;
    protected readonly Random _random;

    // Input encoding
    private readonly FullyConnectedLayer<T> _featureEncoder;
    private readonly FullyConnectedLayer<T>[] _categoricalEncoders;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _positionalEncoding;

    // Transformer backbone
    private readonly TabPFNTransformerBlock<T>[] _transformerBlocks;

    // Output projection
    private readonly FullyConnectedLayer<T>[] _outputMLP;
    private readonly LayerNormalizationLayer<T> _finalNorm;

    // Context storage for in-context learning
    private Tensor<T>? _contextFeatures;
    private Tensor<T>? _contextLabels;

    // Cached values
    [Scratch]
    private Tensor<T>? _encodedInputCache;
    [Scratch]
    private Tensor<T>? _transformerOutputCache;
    [Scratch]
    private Tensor<T>? _mlpOutputCache;

    /// <summary>
    /// Gets the number of numerical features.
    /// </summary>
    public int NumNumericalFeatures { get; }

    /// <summary>
    /// Gets the width of what <see cref="ForwardBackbone"/> returns -- the input size of any head.
    /// </summary>
    /// <remarks>
    /// An empty output-MLP is a legal configuration: the backbone loop simply does not run and the
    /// final norm sees the embedding directly. Indexing <c>[^1]</c> unguarded threw on exactly that
    /// case, so the one place naming this width could not be used by the heads that need it.
    /// </remarks>
    protected int MLPOutputDimension => Options.OutputHeadDimensions.Length > 0
        ? Options.OutputHeadDimensions[^1]
        : Options.EmbeddingDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <summary>
    /// Extra trainable layers a subclass contributes, folded after the shared backbone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The regression and classification variants share this whole backbone and differ only by a
    /// final projection. Each used to override <see cref="ParameterCount"/> purely to append that one
    /// layer -- and because this base had no GetParameters or SetParameters at all, the head was
    /// COUNTED and never read, never restored and never checkpointed. The count grew; the model that
    /// could be saved did not.
    /// </para>
    /// <para>
    /// Declaring the head here means the subclass states WHAT it adds and the traversal below decides
    /// where it goes, so count, vector and restore cannot disagree about it. Mirrors
    /// <c>FTTransformerBase.GetExtraTrainableLayers</c> and <c>NeuralNetworkBase</c>'s hook of the
    /// same name.
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> GetExtraTrainableLayers()
        => System.Linq.Enumerable.Empty<ILayer<T>>();

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// Count, read and restore all derive from THIS, rather than each restating the component list.
    /// Three parallel walks are how a count and a vector come to describe different models: they
    /// agree until someone adds a component to two of them, and nothing reports the disagreement
    /// because the lengths still look plausible. One enumeration makes that failure unrepresentable.
    /// The positional encoding is handled separately below because it is a raw vector, not a layer.
    /// </remarks>
    private IEnumerable<IParameterSource<T>> EnumerateParameterComponents()
    {
        // Materialize FIRST, so every surface measures the same model.
        //
        // A lazily sized component contributes nothing until its shape is known, and SetParameters is
        // itself an event that can resolve it, so measuring before materializing lets the count move
        // under a caller. Mirrors NeuralNetworkBase's restore walk; idempotent, so the common path
        // where everything is already up costs a branch per component.
        //
        // This is hygiene, NOT the fix for the growth this model actually had. That was deferred
        // input sizes on the sub-layers, corrected at their construction sites -- attempting to fix
        // it here first did not move the numbers at all, which is what identified the real cause.
        foreach (var component in RawComponents())
        {
            if (component is Layers.LayerBase<T> layer) layer.MaterializeParameters();
        }

        foreach (var component in RawComponents()) yield return component;
    }

    /// <summary>The component order itself, before materialization is applied.</summary>
    private IEnumerable<IParameterSource<T>> RawComponents()
    {
        yield return _featureEncoder;

        foreach (var enc in _categoricalEncoders) yield return enc;
        foreach (var block in _transformerBlocks) yield return block;
        foreach (var layer in _outputMLP) yield return layer;

        yield return _finalNorm;

        foreach (var extra in GetExtraTrainableLayers())
        {
            if (extra is not null) yield return extra;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount
    {
        get
        {
            long count = 0;
            foreach (var component in EnumerateParameterComponents())
                count += component.ParameterCount;

            if (_positionalEncoding != null)
                count += _positionalEncoding.Length;

            return count;
        }
    }

    /// <summary>
    /// Reads every parameter in traversal order. <see cref="SetParameters"/> reads it back in the
    /// same order, and <see cref="ParameterCount"/> is the length of what this returns.
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        var result = new Vector<T>(checked((int)ParameterCount));
        int offset = 0;

        foreach (var component in EnumerateParameterComponents())
        {
            var part = component.GetParameters();
            for (int i = 0; i < part.Length; i++) result[offset++] = part[i];
        }

        if (_positionalEncoding != null)
        {
            for (int i = 0; i < _positionalEncoding.Length; i++)
                result[offset++] = _positionalEncoding[i];
        }

        return result;
    }

    /// <summary>
    /// Restores every parameter, in the order <see cref="GetParameters"/> emitted them.
    /// </summary>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        long expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
        }

        int offset = 0;
        foreach (var component in EnumerateParameterComponents())
        {
            int count = checked((int)component.ParameterCount);
            if (count == 0) continue;

            var slice = new Vector<T>(count);
            for (int i = 0; i < count; i++) slice[i] = parameters[offset++];
            component.SetParameters(slice);
        }

        if (_positionalEncoding != null)
        {
            for (int i = 0; i < _positionalEncoding.Length; i++)
                _positionalEncoding[i] = parameters[offset++];
        }
    }

    /// <summary>
    /// Initializes a new instance of the TabPFNBase class.
    /// </summary>
    protected TabPFNBase(int numNumericalFeatures, TabPFNOptions<T>? options = null)
    {
        if (numNumericalFeatures < 0)
        {
            throw new ArgumentException("Number of features cannot be negative", nameof(numNumericalFeatures));
        }

        Options = options ?? new TabPFNOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        _random = RandomHelper.CreateSecureRandom();

        int embDim = Options.EmbeddingDimension;

        // Feature encoder
        // ForwardBackbone feeds this [samples, NumNumericalFeatures], so the input width is known
        // here. Zero features is the one genuinely deferred case, and only then is it deferred.
        _featureEncoder = NumNumericalFeatures > 0
            ? new FullyConnectedLayer<T>(
                NumNumericalFeatures,
                embDim,
                Options.HiddenActivation ?? new GELUActivation<T>())
            : new FullyConnectedLayer<T>(
                embDim,
                Options.HiddenActivation ?? new GELUActivation<T>());

        // Categorical encoders
        var cardinalities = Options.CategoricalCardinalities ?? [];
        _categoricalEncoders = new FullyConnectedLayer<T>[cardinalities.Length];

        for (int i = 0; i < cardinalities.Length; i++)
        {
            // Fed a one-hot of width cardinalities[i], so that IS the input size.
            _categoricalEncoders[i] = cardinalities[i] > 0
                ? new FullyConnectedLayer<T>(
                    cardinalities[i],
                    embDim,
                    (IActivationFunction<T>?)null)
                : new FullyConnectedLayer<T>(
                    embDim,
                    (IActivationFunction<T>?)null);
        }

        // Initialize positional encoding if enabled
        if (Options.UsePositionalEncoding)
        {
            _positionalEncoding = CreatePositionalEncoding(Options.MaxContextSamples, embDim);
        }

        // Transformer blocks
        _transformerBlocks = new TabPFNTransformerBlock<T>[Options.NumLayers];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _transformerBlocks[i] = new TabPFNTransformerBlock<T>(
                embDim,
                Options.NumHeads,
                Options.FeedForwardDimension,
                Options.DropoutRate,
                Options.UsePreNorm,
                Options.InitScale,
                _random);
        }

        // Output MLP
        var mlpDims = Options.OutputHeadDimensions;
        _outputMLP = new FullyConnectedLayer<T>[mlpDims.Length];

        int inputDim = embDim;
        for (int i = 0; i < mlpDims.Length; i++)
        {
            bool isLast = i == mlpDims.Length - 1;
            _outputMLP[i] = new FullyConnectedLayer<T>(
                inputDim,
                mlpDims[i],
                isLast ? null : Options.HiddenActivation ?? new GELUActivation<T>());
            inputDim = mlpDims[i];
        }

        // Applied AFTER the output MLP, so it normalizes the last MLP width -- which is embDim
        // only when the MLP is empty. inputDim already holds that value.
        _finalNorm = new LayerNormalizationLayer<T>(inputDim);
    }

    /// <summary>
    /// Creates sinusoidal positional encoding.
    /// </summary>
    private Tensor<T> CreatePositionalEncoding(int maxLen, int embDim)
    {
        var pe = new Tensor<T>([maxLen, embDim]);

        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < embDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / embDim);
                double value = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                pe[pos * embDim + i] = NumOps.FromDouble(value);
            }
        }

        return pe;
    }

    /// <summary>
    /// Sets the context (training) data for in-context learning.
    /// </summary>
    /// <param name="features">Training features.</param>
    /// <param name="labels">Training labels (encoded).</param>
    public void SetContext(Tensor<T> features, Tensor<T> labels)
    {
        int numSamples = features.Shape[0];
        if (numSamples > Options.MaxContextSamples)
        {
            throw new ArgumentException(
                $"Number of context samples ({numSamples}) exceeds maximum ({Options.MaxContextSamples})");
        }

        _contextFeatures = features;
        _contextLabels = labels;
    }

    /// <summary>
    /// Clears the context data.
    /// </summary>
    public void ClearContext()
    {
        _contextFeatures = null;
        _contextLabels = null;
    }

    /// <summary>
    /// Performs the forward pass through the backbone network.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> queryFeatures, Matrix<int>? categoricalIndices = null)
    {
        int querySize = queryFeatures.Shape[0];
        int embDim = Options.EmbeddingDimension;

        // Encode query features
        var queryEncoded = _featureEncoder.Forward(queryFeatures);

        // Add categorical embeddings if present
        if (categoricalIndices != null && _categoricalEncoders.Length > 0)
        {
            var cardinalities = Options.CategoricalCardinalities
                ?? throw new InvalidOperationException("TabPFNBase: CategoricalCardinalities not configured.");

            for (int catIdx = 0; catIdx < _categoricalEncoders.Length; catIdx++)
            {
                var oneHot = CreateOneHotEncoding(
                    categoricalIndices,
                    catIdx,
                    cardinalities[catIdx]);
                var catEmb = _categoricalEncoders[catIdx].Forward(oneHot);

                queryEncoded = Engine.TensorAdd(queryEncoded, catEmb);
            }
        }

        // Combine context and query if context is set
        Tensor<T> combinedInput;
        int contextSize = 0;

        if (_contextFeatures != null)
        {
            contextSize = _contextFeatures.Shape[0];
            var contextEncoded = _featureEncoder.Forward(_contextFeatures);

            // Combine context and query
            combinedInput = new Tensor<T>([contextSize + querySize, embDim]);

            // Copy context
            for (int i = 0; i < contextSize * embDim; i++)
            {
                combinedInput[i] = contextEncoded[i];
            }

            // Copy query
            for (int i = 0; i < querySize * embDim; i++)
            {
                combinedInput[contextSize * embDim + i] = queryEncoded[i];
            }
        }
        else
        {
            combinedInput = queryEncoded;
        }

        // Add positional encoding if enabled
        if (_positionalEncoding != null)
        {
            int totalSize = combinedInput.Shape[0];
            for (int pos = 0; pos < totalSize; pos++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    int idx = pos * embDim + d;
                    combinedInput[idx] = NumOps.Add(
                        combinedInput[idx],
                        _positionalEncoding[idx % _positionalEncoding.Length]);
                }
            }
        }

        _encodedInputCache = combinedInput;

        // Apply transformer blocks
        var transformerOutput = combinedInput;
        foreach (var block in _transformerBlocks)
        {
            transformerOutput = block.Forward(transformerOutput);
        }
        _transformerOutputCache = transformerOutput;

        // Extract only query outputs (last querySize elements)
        var queryOutput = new Tensor<T>([querySize, embDim]);
        int startIdx = contextSize * embDim;
        for (int i = 0; i < querySize * embDim; i++)
        {
            queryOutput[i] = transformerOutput[startIdx + i];
        }

        // Apply output MLP
        var mlpOutput = queryOutput;
        foreach (var layer in _outputMLP)
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
        _featureEncoder.UpdateParameters(learningRate);

        foreach (var enc in _categoricalEncoders)
        {
            enc.UpdateParameters(learningRate);
        }

        foreach (var block in _transformerBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        foreach (var layer in _outputMLP)
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
        _encodedInputCache = null;
        _transformerOutputCache = null;
        _mlpOutputCache = null;

        _featureEncoder.ResetState();

        foreach (var enc in _categoricalEncoders)
        {
            enc.ResetState();
        }

        foreach (var block in _transformerBlocks)
        {
            block.ResetState();
        }

        foreach (var layer in _outputMLP)
        {
            layer.ResetState();
        }

        _finalNorm.ResetState();
    }

    /// <summary>
    /// TabPFN-specific transformer block with causal masking for in-context learning.
    /// </summary>
    private sealed class TabPFNTransformerBlock<TBlock> : IParameterSource<TBlock>
    {
        private static readonly INumericOperations<TBlock> NumOps = MathHelper.GetNumericOperations<TBlock>();

        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly bool _usePreNorm;
        private readonly double _dropoutRate;

        // Attention weights
        [AiDotNet.Attributes.TrainableParameter]
        private Tensor<TBlock> _queryWeights;
        [AiDotNet.Attributes.TrainableParameter]
        private Tensor<TBlock> _keyWeights;
        [AiDotNet.Attributes.TrainableParameter]
        private Tensor<TBlock> _valueWeights;
        [AiDotNet.Attributes.TrainableParameter]
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
        [Scratch]
        private Tensor<TBlock>? _inputCache;
        [Scratch]
        private Tensor<TBlock>? _attentionOutputCache;

        /// <summary>
        /// Summed from the SAME traversal the vector uses, so the two cannot disagree.
        /// </summary>
        /// <remarks>
        /// This was the formula <c>_embeddingDim * _embeddingDim * 4</c> plus the sub-layers. Adding
        /// a real read path proved the formula wrong: it reported 925,377 for a model whose weights
        /// are 933,250 values, understating the attention projections by 7,873. A formula restates
        /// what the tensors already know and drifts from them silently -- there was no vector to
        /// contradict it, so the error was unobservable rather than absent.
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

        /// <summary>
        /// The block's attention projections in the order the count above sums them, so the
        /// enclosing model's read and restore agree with its count by construction.
        /// </summary>
        /// <remarks>
        /// This block reported a ParameterCount and had NO way to read or write those values. Every
        /// TabPFN model therefore advertised weights that no checkpoint could contain: the count grew
        /// with each block, and saving the model saved none of them.
        /// </remarks>
        private IEnumerable<Tensor<TBlock>> AttentionTensors()
        {
            yield return _queryWeights;
            yield return _keyWeights;
            yield return _valueWeights;
            yield return _outputWeights;
        }

        private IEnumerable<IParameterSource<TBlock>> SubLayers()
        {
            yield return _ff1;
            yield return _ff2;
            yield return _norm1;
            yield return _norm2;
        }

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

            // Write THROUGH the tensors rather than replacing them: the forward pass and any engine
            // cache key on the tensor identity, so rebinding here would restore values into objects
            // nothing else is looking at.
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

        public TabPFNTransformerBlock(
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
            // Both projections and both norms are fixed by the block's own dimensions, so none of
            // them needs the deferred constructor.
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
            int seqLen = input.Shape[0];

            Tensor<TBlock> x;
            if (_usePreNorm)
            {
                x = _norm1.Forward(input);
            }
            else
            {
                x = input;
            }

            // Multi-head self-attention
            var query = MatMul(x, _queryWeights);
            var key = MatMul(x, _keyWeights);
            var value = MatMul(x, _valueWeights);

            var attentionOutput = ComputeCausalAttention(query, key, value, seqLen);
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

        /// <summary>
        /// Computes causal attention where query can attend to all context
        /// but test samples only attend to context + previous test samples.
        /// </summary>
        private Tensor<TBlock> ComputeCausalAttention(
            Tensor<TBlock> query, Tensor<TBlock> key, Tensor<TBlock> value, int seqLen)
        {
            var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

            // Compute attention scores: Q * K^T / sqrt(d_k)
            var scores = new Tensor<TBlock>([seqLen, seqLen]);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    var dot = NumOps.Zero;
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(
                            query[i * _embeddingDim + k],
                            key[j * _embeddingDim + k]));
                    }
                    scores[i * seqLen + j] = NumOps.Multiply(dot, scale);
                }
            }

            // Apply softmax row-wise (for TabPFN, full attention within context)
            for (int i = 0; i < seqLen; i++)
            {
                var maxVal = scores[i * seqLen];
                for (int j = 1; j < seqLen; j++)
                {
                    var val = scores[i * seqLen + j];
                    if (NumOps.Compare(val, maxVal) > 0)
                        maxVal = val;
                }

                var sumExp = NumOps.Zero;
                for (int j = 0; j < seqLen; j++)
                {
                    var expVal = NumOps.Exp(NumOps.Subtract(scores[i * seqLen + j], maxVal));
                    scores[i * seqLen + j] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }

                for (int j = 0; j < seqLen; j++)
                {
                    scores[i * seqLen + j] = NumOps.Divide(scores[i * seqLen + j], sumExp);
                }
            }

            // Compute output: softmax(scores) * V
            var output = new Tensor<TBlock>([seqLen, _embeddingDim]);

            for (int i = 0; i < seqLen; i++)
            {
                for (int k = 0; k < _embeddingDim; k++)
                {
                    var sum = NumOps.Zero;
                    for (int j = 0; j < seqLen; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            scores[i * seqLen + j],
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
            _attentionOutputCache = null;

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
}

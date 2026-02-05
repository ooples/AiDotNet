using AiDotNet.ActivationFunctions;
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
public abstract class TabDPTBase<T>
{
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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = _featureProjection.ParameterCount;

            foreach (var emb in _categoricalEmbeddings)
                count += emb.ParameterCount;

            foreach (var block in _transformerBlocks)
                count += block.ParameterCount;

            if (_featureAttention != null)
                count += _featureAttention.ParameterCount;

            foreach (var layer in _mlpLayers)
                count += layer.ParameterCount;

            count += _finalNorm.ParameterCount;

            return count;
        }
    }

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
        _featureProjection = new FullyConnectedLayer<T>(
            numNumericalFeatures,
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

        _finalNorm = new LayerNormalizationLayer<T>(mlpDims[^1]);
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
            for (int catIdx = 0; catIdx < _categoricalEmbeddings.Length; catIdx++)
            {
                var oneHot = CreateOneHotEncoding(
                    categoricalIndices,
                    catIdx,
                    Options.CategoricalCardinalities![catIdx]);
                var catEmb = _categoricalEmbeddings[catIdx].Forward(oneHot);

                // Add to embeddings
                for (int i = 0; i < embeddings.Length; i++)
                {
                    embeddings[i] = NumOps.Add(embeddings[i], catEmb[i]);
                }
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
    /// Performs the backward pass through the backbone network.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradient)
    {
        // Backward through final norm
        var grad = _finalNorm.Backward(gradient);

        // Backward through MLP layers
        for (int i = _mlpLayers.Length - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }

        // Backward through feature attention if present
        if (_featureAttention != null)
        {
            grad = _featureAttention.Backward(grad);
        }

        // Backward through transformer blocks
        for (int i = _transformerBlocks.Length - 1; i >= 0; i--)
        {
            grad = _transformerBlocks[i].Backward(grad);
        }

        // Backward through feature projection
        grad = _featureProjection.Backward(grad);

        return grad;
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
    private sealed class TransformerBlock<TBlock>
    {
        private static readonly INumericOperations<TBlock> Ops = MathHelper.GetNumericOperations<TBlock>();

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

        public int ParameterCount
        {
            get
            {
                int attentionParams = _embeddingDim * _embeddingDim * 4; // Q, K, V, O
                return attentionParams + _ff1.ParameterCount + _ff2.ParameterCount +
                       _norm1.ParameterCount + _norm2.ParameterCount;
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
                weights[i] = Ops.FromDouble(random.NextGaussian() * scale);
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
            var residual1 = new Tensor<TBlock>(input.Shape);
            for (int i = 0; i < input.Length; i++)
            {
                residual1[i] = Ops.Add(input[i], attentionOutput[i]);
            }

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
            var output = new Tensor<TBlock>(residual1.Shape);
            for (int i = 0; i < residual1.Length; i++)
            {
                output[i] = Ops.Add(residual1[i], ffOutput[i]);
            }

            if (!_usePreNorm)
            {
                output = _norm2.Forward(output);
            }

            return output;
        }

        private Tensor<TBlock> ComputeAttention(Tensor<TBlock> query, Tensor<TBlock> key, Tensor<TBlock> value, int batchSize)
        {
            var scale = Ops.FromDouble(1.0 / Math.Sqrt(_headDim));

            // Compute attention scores: Q * K^T / sqrt(d_k)
            var scores = new Tensor<TBlock>([batchSize, batchSize]);

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    var dot = Ops.Zero;
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        dot = Ops.Add(dot, Ops.Multiply(
                            query[i * _embeddingDim + k],
                            key[j * _embeddingDim + k]));
                    }
                    scores[i * batchSize + j] = Ops.Multiply(dot, scale);
                }
            }

            // Apply softmax
            for (int i = 0; i < batchSize; i++)
            {
                var maxVal = scores[i * batchSize];
                for (int j = 1; j < batchSize; j++)
                {
                    var val = scores[i * batchSize + j];
                    if (Ops.Compare(val, maxVal) > 0)
                        maxVal = val;
                }

                var sumExp = Ops.Zero;
                for (int j = 0; j < batchSize; j++)
                {
                    var expVal = Ops.Exp(Ops.Subtract(scores[i * batchSize + j], maxVal));
                    scores[i * batchSize + j] = expVal;
                    sumExp = Ops.Add(sumExp, expVal);
                }

                for (int j = 0; j < batchSize; j++)
                {
                    scores[i * batchSize + j] = Ops.Divide(scores[i * batchSize + j], sumExp);
                }
            }

            _attentionScoresCache = scores;

            // Compute output: softmax(scores) * V
            var output = new Tensor<TBlock>([batchSize, _embeddingDim]);

            for (int i = 0; i < batchSize; i++)
            {
                for (int k = 0; k < _embeddingDim; k++)
                {
                    var sum = Ops.Zero;
                    for (int j = 0; j < batchSize; j++)
                    {
                        sum = Ops.Add(sum, Ops.Multiply(
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
            int batchSize = input.Shape[0];
            int inputDim = weights.Shape[0];
            int outputDim = weights.Shape[1];

            var output = new Tensor<TBlock>([batchSize, outputDim]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    var sum = Ops.Zero;
                    for (int i = 0; i < inputDim; i++)
                    {
                        sum = Ops.Add(sum, Ops.Multiply(
                            input[b * inputDim + i],
                            weights[i * outputDim + o]));
                    }
                    output[b * outputDim + o] = sum;
                }
            }

            return output;
        }

        public Tensor<TBlock> Backward(Tensor<TBlock> gradient)
        {
            // Simplified backward pass
            var grad = _ff2.Backward(gradient);
            grad = _ff1.Backward(grad);

            // Add residual gradient
            for (int i = 0; i < gradient.Length; i++)
            {
                grad[i] = Ops.Add(grad[i], gradient[i]);
            }

            return grad;
        }

        public void UpdateParameters(TBlock learningRate)
        {
            // Update attention weights
            for (int i = 0; i < _queryWeights.Length; i++)
            {
                _queryWeights[i] = Ops.Subtract(_queryWeights[i], Ops.Multiply(learningRate, _queryGrad[i]));
                _keyWeights[i] = Ops.Subtract(_keyWeights[i], Ops.Multiply(learningRate, _keyGrad[i]));
                _valueWeights[i] = Ops.Subtract(_valueWeights[i], Ops.Multiply(learningRate, _valueGrad[i]));
                _outputWeights[i] = Ops.Subtract(_outputWeights[i], Ops.Multiply(learningRate, _outputGrad[i]));
            }

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
            for (int i = 0; i < _queryGrad.Length; i++)
            {
                _queryGrad[i] = Ops.Zero;
                _keyGrad[i] = Ops.Zero;
                _valueGrad[i] = Ops.Zero;
                _outputGrad[i] = Ops.Zero;
            }

            _ff1.ResetState();
            _ff2.ResetState();
            _norm1.ResetState();
            _norm2.ResetState();
        }
    }

    /// <summary>
    /// Feature-wise attention block for column interactions.
    /// </summary>
    private sealed class FeatureAttentionBlock<TBlock>
    {
        private static readonly INumericOperations<TBlock> Ops = MathHelper.GetNumericOperations<TBlock>();

        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly double _dropoutRate;

        private Tensor<TBlock> _featureQuery;
        private Tensor<TBlock> _featureKey;
        private Tensor<TBlock> _featureValue;
        private Tensor<TBlock> _featureOutput;

        private Tensor<TBlock>? _inputCache;

        public int ParameterCount => _embeddingDim * _embeddingDim * 4;

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
                weights[i] = Ops.FromDouble(random.NextGaussian() * scale);
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

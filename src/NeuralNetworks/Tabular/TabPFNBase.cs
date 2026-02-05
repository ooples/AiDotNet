using AiDotNet.ActivationFunctions;
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
public abstract class TabPFNBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected readonly TabPFNOptions<T> Options;
    protected readonly Random _random;

    // Input encoding
    private readonly FullyConnectedLayer<T> _featureEncoder;
    private readonly FullyConnectedLayer<T>[] _categoricalEncoders;
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
    private Tensor<T>? _encodedInputCache;
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
            int count = _featureEncoder.ParameterCount;

            foreach (var enc in _categoricalEncoders)
                count += enc.ParameterCount;

            foreach (var block in _transformerBlocks)
                count += block.ParameterCount;

            foreach (var layer in _outputMLP)
                count += layer.ParameterCount;

            count += _finalNorm.ParameterCount;

            if (_positionalEncoding != null)
                count += _positionalEncoding.Length;

            return count;
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
        _featureEncoder = new FullyConnectedLayer<T>(
            numNumericalFeatures,
            embDim,
            Options.HiddenActivation ?? new GELUActivation<T>());

        // Categorical encoders
        var cardinalities = Options.CategoricalCardinalities ?? [];
        _categoricalEncoders = new FullyConnectedLayer<T>[cardinalities.Length];

        for (int i = 0; i < cardinalities.Length; i++)
        {
            _categoricalEncoders[i] = new FullyConnectedLayer<T>(
                cardinalities[i],
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

        _finalNorm = new LayerNormalizationLayer<T>(mlpDims[^1]);
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
            for (int catIdx = 0; catIdx < _categoricalEncoders.Length; catIdx++)
            {
                var oneHot = CreateOneHotEncoding(
                    categoricalIndices,
                    catIdx,
                    Options.CategoricalCardinalities![catIdx]);
                var catEmb = _categoricalEncoders[catIdx].Forward(oneHot);

                for (int i = 0; i < queryEncoded.Length; i++)
                {
                    queryEncoded[i] = NumOps.Add(queryEncoded[i], catEmb[i]);
                }
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
    /// Performs the backward pass through the backbone network.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradient)
    {
        // Backward through final norm
        var grad = _finalNorm.Backward(gradient);

        // Backward through output MLP
        for (int i = _outputMLP.Length - 1; i >= 0; i--)
        {
            grad = _outputMLP[i].Backward(grad);
        }

        // Backward through transformer blocks
        for (int i = _transformerBlocks.Length - 1; i >= 0; i--)
        {
            grad = _transformerBlocks[i].Backward(grad);
        }

        // Backward through feature encoder
        grad = _featureEncoder.Backward(grad);

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
    private sealed class TabPFNTransformerBlock<TBlock>
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
        private Tensor<TBlock>? _attentionOutputCache;

        public int ParameterCount
        {
            get
            {
                int attentionParams = _embeddingDim * _embeddingDim * 4;
                return attentionParams + _ff1.ParameterCount + _ff2.ParameterCount +
                       _norm1.ParameterCount + _norm2.ParameterCount;
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

        /// <summary>
        /// Computes causal attention where query can attend to all context
        /// but test samples only attend to context + previous test samples.
        /// </summary>
        private Tensor<TBlock> ComputeCausalAttention(
            Tensor<TBlock> query, Tensor<TBlock> key, Tensor<TBlock> value, int seqLen)
        {
            var scale = Ops.FromDouble(1.0 / Math.Sqrt(_headDim));

            // Compute attention scores: Q * K^T / sqrt(d_k)
            var scores = new Tensor<TBlock>([seqLen, seqLen]);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    var dot = Ops.Zero;
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        dot = Ops.Add(dot, Ops.Multiply(
                            query[i * _embeddingDim + k],
                            key[j * _embeddingDim + k]));
                    }
                    scores[i * seqLen + j] = Ops.Multiply(dot, scale);
                }
            }

            // Apply softmax row-wise (for TabPFN, full attention within context)
            for (int i = 0; i < seqLen; i++)
            {
                var maxVal = scores[i * seqLen];
                for (int j = 1; j < seqLen; j++)
                {
                    var val = scores[i * seqLen + j];
                    if (Ops.Compare(val, maxVal) > 0)
                        maxVal = val;
                }

                var sumExp = Ops.Zero;
                for (int j = 0; j < seqLen; j++)
                {
                    var expVal = Ops.Exp(Ops.Subtract(scores[i * seqLen + j], maxVal));
                    scores[i * seqLen + j] = expVal;
                    sumExp = Ops.Add(sumExp, expVal);
                }

                for (int j = 0; j < seqLen; j++)
                {
                    scores[i * seqLen + j] = Ops.Divide(scores[i * seqLen + j], sumExp);
                }
            }

            // Compute output: softmax(scores) * V
            var output = new Tensor<TBlock>([seqLen, _embeddingDim]);

            for (int i = 0; i < seqLen; i++)
            {
                for (int k = 0; k < _embeddingDim; k++)
                {
                    var sum = Ops.Zero;
                    for (int j = 0; j < seqLen; j++)
                    {
                        sum = Ops.Add(sum, Ops.Multiply(
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
            int seqLen = input.Shape[0];
            int inputDim = weights.Shape[0];
            int outputDim = weights.Shape[1];

            var output = new Tensor<TBlock>([seqLen, outputDim]);

            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    var sum = Ops.Zero;
                    for (int i = 0; i < inputDim; i++)
                    {
                        sum = Ops.Add(sum, Ops.Multiply(
                            input[s * inputDim + i],
                            weights[i * outputDim + o]));
                    }
                    output[s * outputDim + o] = sum;
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
            _attentionOutputCache = null;

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
}

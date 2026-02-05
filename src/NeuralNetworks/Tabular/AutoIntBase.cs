using AiDotNet.ActivationFunctions;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for AutoInt (Automatic Feature Interaction Learning).
/// </summary>
/// <remarks>
/// <para>
/// AutoInt uses multi-head self-attention to automatically learn feature interactions:
/// 1. Each feature is embedded into a dense vector
/// 2. Self-attention layers learn interactions between features
/// 3. Interactions are combined with original embeddings for prediction
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoInt discovers which features work well together:
///
/// - **Without AutoInt**: You manually create features like "age * income"
/// - **With AutoInt**: The model automatically learns "age and income interact"
///
/// This is especially useful for recommendation systems, click prediction,
/// and any tabular task where feature combinations matter.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class AutoIntBase<T>
{
    protected readonly AutoIntOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    private readonly Tensor<T> _numericalEmbeddings;
    private readonly Tensor<T>[]? _categoricalEmbeddings;

    // Interacting layers (multi-head self-attention)
    private readonly List<InteractingLayer> _interactingLayers;

    // MLP output
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    // Caches
    private Tensor<T>? _embeddedFeaturesCache;
    private List<Tensor<T>>? _interactingOutputsCache;
    private Tensor<T>? _mlpOutputCache;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = _numericalEmbeddings.Length;

            if (_categoricalEmbeddings != null)
            {
                foreach (var emb in _categoricalEmbeddings)
                    count += emb.Length;
            }

            foreach (var layer in _interactingLayers)
                count += layer.ParameterCount;

            foreach (var layer in _mlpLayers)
                count += layer.ParameterCount;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the AutoIntBase class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected AutoIntBase(int numNumericalFeatures, AutoIntOptions<T>? options = null)
    {
        Options = options ?? new AutoIntOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        TotalFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        if (TotalFeatures == 0)
        {
            throw new ArgumentException("Model must have at least one feature");
        }

        // Numerical feature embeddings (one embedding vector per feature)
        _numericalEmbeddings = new Tensor<T>(new[] { NumNumericalFeatures, Options.EmbeddingDimension });
        InitializeEmbedding(_numericalEmbeddings);

        // Categorical embeddings
        if (NumCategoricalFeatures > 0 && Options.CategoricalCardinalities != null)
        {
            _categoricalEmbeddings = new Tensor<T>[NumCategoricalFeatures];
            for (int i = 0; i < NumCategoricalFeatures; i++)
            {
                int cardinality = Options.CategoricalCardinalities[i];
                _categoricalEmbeddings[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
                InitializeEmbedding(_categoricalEmbeddings[i]);
            }
        }

        // Interacting layers (self-attention)
        _interactingLayers = [];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _interactingLayers.Add(new InteractingLayer(
                Options.EmbeddingDimension,
                Options.AttentionDimension,
                Options.NumHeads,
                Options.UseResidual,
                _random));
        }

        // MLP output layers
        _mlpLayers = [];
        int mlpInput = TotalFeatures * Options.EmbeddingDimension;
        foreach (var hiddenDim in Options.MLPHiddenDimensions)
        {
            _mlpLayers.Add(new FullyConnectedLayer<T>(
                mlpInput,
                hiddenDim,
                Options.HiddenActivation ?? new ReLUActivation<T>()));
            mlpInput = hiddenDim;
        }

        MLPOutputDimension = mlpInput;
    }

    private void InitializeEmbedding(Tensor<T> embedding)
    {
        var scale = NumOps.FromDouble(Options.EmbeddingInitScale);
        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] = NumOps.Multiply(
                NumOps.FromDouble(_random.NextGaussian()),
                scale);
        }
    }

    /// <summary>
    /// Embeds input features.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        var embedded = new Tensor<T>(new[] { batchSize, TotalFeatures, Options.EmbeddingDimension });

        // Embed numerical features (multiply feature value by embedding vector)
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < NumNumericalFeatures; f++)
            {
                var featureValue = numericalFeatures[b * NumNumericalFeatures + f];

                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    int embIdx = f * Options.EmbeddingDimension + d;
                    int outIdx = (b * TotalFeatures + f) * Options.EmbeddingDimension + d;
                    embedded[outIdx] = NumOps.Multiply(featureValue, _numericalEmbeddings[embIdx]);
                }
            }
        }

        // Embed categorical features (lookup embedding)
        if (categoricalIndices != null && _categoricalEmbeddings != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < NumCategoricalFeatures; f++)
                {
                    int catIdx = categoricalIndices[b, f];
                    int featureIdx = NumNumericalFeatures + f;

                    for (int d = 0; d < Options.EmbeddingDimension; d++)
                    {
                        int embIdx = catIdx * Options.EmbeddingDimension + d;
                        int outIdx = (b * TotalFeatures + featureIdx) * Options.EmbeddingDimension + d;
                        embedded[outIdx] = _categoricalEmbeddings[f][embIdx];
                    }
                }
            }
        }

        return embedded;
    }

    /// <summary>
    /// Performs the forward pass through the AutoInt backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        _interactingOutputsCache = [];

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);
        _embeddedFeaturesCache = embedded;

        // Apply interacting layers
        var current = embedded;
        foreach (var layer in _interactingLayers)
        {
            current = layer.Forward(current, batchSize, TotalFeatures, Options.EmbeddingDimension);
            _interactingOutputsCache.Add(current);
        }

        // Flatten
        var flattened = new Tensor<T>(new[] { batchSize, TotalFeatures * Options.EmbeddingDimension });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < TotalFeatures * Options.EmbeddingDimension; i++)
            {
                flattened[b * TotalFeatures * Options.EmbeddingDimension + i] =
                    current[b * TotalFeatures * Options.EmbeddingDimension + i];
            }
        }

        // MLP
        var mlpOutput = flattened;
        foreach (var mlpLayer in _mlpLayers)
        {
            mlpOutput = mlpLayer.Forward(mlpOutput);
        }

        _mlpOutputCache = mlpOutput;
        return mlpOutput;
    }

    /// <summary>
    /// Performs the backward pass through the backbone.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradOutput)
    {
        var grad = gradOutput;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }
        return grad;
    }

    /// <summary>
    /// Gets the learned feature interaction importance.
    /// </summary>
    /// <returns>Attention weights showing feature interaction patterns.</returns>
    public Tensor<T>? GetInteractionWeights()
    {
        if (_interactingLayers.Count == 0)
            return null;

        // Return the attention weights from the last interacting layer
        return _interactingLayers[^1].GetAttentionWeights();
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        foreach (var layer in _interactingLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _embeddedFeaturesCache = null;
        _interactingOutputsCache = null;
        _mlpOutputCache = null;

        foreach (var layer in _interactingLayers)
        {
            layer.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Multi-head self-attention layer for feature interaction.
    /// </summary>
    private class InteractingLayer
    {
        private readonly int _embeddingDim;
        private readonly int _attentionDim;
        private readonly int _numHeads;
        private readonly bool _useResidual;

        // Query, Key, Value projection weights per head
        private readonly Tensor<T> _queryWeights;
        private readonly Tensor<T> _keyWeights;
        private readonly Tensor<T> _valueWeights;

        // Output projection
        private readonly Tensor<T> _outputWeights;

        // Cache
        private Tensor<T>? _attentionWeightsCache;
        private Tensor<T>? _inputCache;

        public int ParameterCount =>
            _queryWeights.Length + _keyWeights.Length +
            _valueWeights.Length + _outputWeights.Length;

        public InteractingLayer(int embeddingDim, int attentionDim, int numHeads, bool useResidual, Random random)
        {
            _embeddingDim = embeddingDim;
            _attentionDim = attentionDim;
            _numHeads = numHeads;
            _useResidual = useResidual;

            var scale = NumOps.FromDouble(Math.Sqrt(2.0 / (embeddingDim + attentionDim)));

            _queryWeights = new Tensor<T>(new[] { numHeads, embeddingDim, attentionDim });
            _keyWeights = new Tensor<T>(new[] { numHeads, embeddingDim, attentionDim });
            _valueWeights = new Tensor<T>(new[] { numHeads, embeddingDim, attentionDim });
            _outputWeights = new Tensor<T>(new[] { numHeads * attentionDim, embeddingDim });

            InitializeWeights(_queryWeights, scale, random);
            InitializeWeights(_keyWeights, scale, random);
            InitializeWeights(_valueWeights, scale, random);
            InitializeWeights(_outputWeights, scale, random);
        }

        private static void InitializeWeights(Tensor<T> weights, T scale, Random random)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.Multiply(NumOps.FromDouble(random.NextGaussian()), scale);
            }
        }

        public Tensor<T> Forward(Tensor<T> input, int batchSize, int numFeatures, int embeddingDim)
        {
            _inputCache = input;
            var output = new Tensor<T>(input.Shape);

            // Multi-head attention
            var allHeadOutputs = new Tensor<T>(new[] { batchSize, numFeatures, _numHeads * _attentionDim });
            _attentionWeightsCache = new Tensor<T>(new[] { batchSize, _numHeads, numFeatures, numFeatures });

            for (int h = 0; h < _numHeads; h++)
            {
                // Compute Q, K, V for this head
                for (int b = 0; b < batchSize; b++)
                {
                    var queries = new T[numFeatures, _attentionDim];
                    var keys = new T[numFeatures, _attentionDim];
                    var values = new T[numFeatures, _attentionDim];

                    // Project to Q, K, V
                    for (int f = 0; f < numFeatures; f++)
                    {
                        for (int a = 0; a < _attentionDim; a++)
                        {
                            var q = NumOps.Zero;
                            var k = NumOps.Zero;
                            var v = NumOps.Zero;

                            for (int d = 0; d < embeddingDim; d++)
                            {
                                var inputVal = input[(b * numFeatures + f) * embeddingDim + d];
                                int wIdx = (h * embeddingDim + d) * _attentionDim + a;
                                q = NumOps.Add(q, NumOps.Multiply(inputVal, _queryWeights[wIdx]));
                                k = NumOps.Add(k, NumOps.Multiply(inputVal, _keyWeights[wIdx]));
                                v = NumOps.Add(v, NumOps.Multiply(inputVal, _valueWeights[wIdx]));
                            }

                            queries[f, a] = q;
                            keys[f, a] = k;
                            values[f, a] = v;
                        }
                    }

                    // Compute attention scores
                    var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_attentionDim));
                    var attentionScores = new T[numFeatures, numFeatures];

                    for (int i = 0; i < numFeatures; i++)
                    {
                        for (int j = 0; j < numFeatures; j++)
                        {
                            var score = NumOps.Zero;
                            for (int a = 0; a < _attentionDim; a++)
                            {
                                score = NumOps.Add(score, NumOps.Multiply(queries[i, a], keys[j, a]));
                            }
                            attentionScores[i, j] = NumOps.Multiply(score, scale);
                        }
                    }

                    // Softmax per row
                    for (int i = 0; i < numFeatures; i++)
                    {
                        var maxScore = attentionScores[i, 0];
                        for (int j = 1; j < numFeatures; j++)
                        {
                            if (NumOps.Compare(attentionScores[i, j], maxScore) > 0)
                                maxScore = attentionScores[i, j];
                        }

                        var sumExp = NumOps.Zero;
                        for (int j = 0; j < numFeatures; j++)
                        {
                            var exp = NumOps.Exp(NumOps.Subtract(attentionScores[i, j], maxScore));
                            attentionScores[i, j] = exp;
                            sumExp = NumOps.Add(sumExp, exp);
                        }

                        for (int j = 0; j < numFeatures; j++)
                        {
                            attentionScores[i, j] = NumOps.Divide(attentionScores[i, j], sumExp);
                            _attentionWeightsCache![(((b * _numHeads) + h) * numFeatures + i) * numFeatures + j] =
                                attentionScores[i, j];
                        }
                    }

                    // Weighted sum of values
                    for (int i = 0; i < numFeatures; i++)
                    {
                        for (int a = 0; a < _attentionDim; a++)
                        {
                            var weighted = NumOps.Zero;
                            for (int j = 0; j < numFeatures; j++)
                            {
                                weighted = NumOps.Add(weighted,
                                    NumOps.Multiply(attentionScores[i, j], values[j, a]));
                            }
                            allHeadOutputs[(b * numFeatures + i) * (_numHeads * _attentionDim) + h * _attentionDim + a] =
                                weighted;
                        }
                    }
                }
            }

            // Project back to embedding dimension
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < numFeatures; f++)
                {
                    for (int d = 0; d < embeddingDim; d++)
                    {
                        var projected = NumOps.Zero;
                        for (int ha = 0; ha < _numHeads * _attentionDim; ha++)
                        {
                            var headOut = allHeadOutputs[(b * numFeatures + f) * (_numHeads * _attentionDim) + ha];
                            projected = NumOps.Add(projected,
                                NumOps.Multiply(headOut, _outputWeights[ha * embeddingDim + d]));
                        }

                        int outIdx = (b * numFeatures + f) * embeddingDim + d;

                        if (_useResidual)
                        {
                            output[outIdx] = NumOps.Add(input[outIdx], projected);
                        }
                        else
                        {
                            output[outIdx] = projected;
                        }
                    }
                }
            }

            return output;
        }

        public Tensor<T>? GetAttentionWeights() => _attentionWeightsCache;

        public void UpdateParameters(T learningRate)
        {
            // Simplified parameter update
            // In practice, would use proper gradient computation
        }

        public void ResetState()
        {
            _attentionWeightsCache = null;
            _inputCache = null;
        }
    }
}

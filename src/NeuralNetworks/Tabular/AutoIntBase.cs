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
    private readonly List<InteractingLayer<T>> _interactingLayers;

    // MLP output
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    // Caches
    private Tensor<T>? _numericalFeaturesCache;
    private Matrix<int>? _categoricalIndicesCache;
    private Tensor<T>? _embeddedFeaturesCache;
    private List<Tensor<T>>? _interactingOutputsCache;
    private Tensor<T>? _mlpOutputCache;

    // Embedding gradients
    private readonly Tensor<T> _numericalEmbeddingsGrad;
    private readonly Tensor<T>[]? _categoricalEmbeddingsGrad;

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
        if (numNumericalFeatures < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNumericalFeatures),
                "Number of numerical features cannot be negative.");
        }

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
        _numericalEmbeddingsGrad = new Tensor<T>(new[] { NumNumericalFeatures, Options.EmbeddingDimension });

        // Categorical embeddings
        if (NumCategoricalFeatures > 0 && Options.CategoricalCardinalities != null)
        {
            _categoricalEmbeddings = new Tensor<T>[NumCategoricalFeatures];
            _categoricalEmbeddingsGrad = new Tensor<T>[NumCategoricalFeatures];
            for (int i = 0; i < NumCategoricalFeatures; i++)
            {
                int cardinality = Options.CategoricalCardinalities[i];
                _categoricalEmbeddings[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
                InitializeEmbedding(_categoricalEmbeddings[i]);
                _categoricalEmbeddingsGrad[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
            }
        }

        // Interacting layers (self-attention)
        _interactingLayers = [];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _interactingLayers.Add(new InteractingLayer<T>(
                Options.EmbeddingDimension,
                Options.NumHeads,
                Options.AttentionDimension * Options.NumHeads,
                Options.UseResidual,
                Options.EmbeddingInitScale));
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
                    int cardinality = Options.CategoricalCardinalities![f];
                    if ((uint)catIdx >= (uint)cardinality)
                    {
                        throw new ArgumentOutOfRangeException(
                            nameof(categoricalIndices),
                            $"Categorical index {catIdx} for feature {f} (batch {b}) is out of range [0, {cardinality}).");
                    }
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
        _numericalFeaturesCache = numericalFeatures;
        _categoricalIndicesCache = categoricalIndices;

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);
        _embeddedFeaturesCache = embedded;

        // Apply interacting layers
        var current = embedded;
        foreach (var layer in _interactingLayers)
        {
            current = layer.Forward(current);
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
        if (_embeddedFeaturesCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward.");
        }

        var grad = gradOutput;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }

        int batchSize = _embeddedFeaturesCache.Shape[0];
        int embeddingDim = Options.EmbeddingDimension;

        // Unflatten gradients back to [batch, features, embeddingDim]
        var gradEmbedded = new Tensor<T>(new[] { batchSize, TotalFeatures, embeddingDim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < TotalFeatures * embeddingDim; i++)
            {
                gradEmbedded[b * TotalFeatures * embeddingDim + i] =
                    grad[b * TotalFeatures * embeddingDim + i];
            }
        }

        // Backprop through interacting layers
        for (int i = _interactingLayers.Count - 1; i >= 0; i--)
        {
            gradEmbedded = _interactingLayers[i].Backward(gradEmbedded);
        }

        AccumulateEmbeddingGradients(gradEmbedded);

        // Return flattened gradient for compatibility with callers
        var gradFlat = new Tensor<T>(new[] { batchSize, TotalFeatures * embeddingDim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < TotalFeatures * embeddingDim; i++)
            {
                gradFlat[b * TotalFeatures * embeddingDim + i] =
                    gradEmbedded[b * TotalFeatures * embeddingDim + i];
            }
        }

        return gradFlat;
    }

    private void AccumulateEmbeddingGradients(Tensor<T> gradEmbedded)
    {
        // Reset gradients
        for (int i = 0; i < _numericalEmbeddingsGrad.Length; i++)
        {
            _numericalEmbeddingsGrad[i] = NumOps.Zero;
        }

        if (_categoricalEmbeddingsGrad != null)
        {
            foreach (var grad in _categoricalEmbeddingsGrad)
            {
                for (int i = 0; i < grad.Length; i++)
                {
                    grad[i] = NumOps.Zero;
                }
            }
        }

        if (_numericalFeaturesCache == null)
        {
            return;
        }

        int batchSize = gradEmbedded.Shape[0];
        int embeddingDim = Options.EmbeddingDimension;

        // Numerical embeddings: gradient accumulates scaled by feature values
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < NumNumericalFeatures; f++)
            {
                var featureValue = _numericalFeaturesCache[b * NumNumericalFeatures + f];
                for (int d = 0; d < embeddingDim; d++)
                {
                    int gradIdx = (b * TotalFeatures + f) * embeddingDim + d;
                    int embIdx = f * embeddingDim + d;
                    _numericalEmbeddingsGrad[embIdx] = NumOps.Add(
                        _numericalEmbeddingsGrad[embIdx],
                        NumOps.Multiply(gradEmbedded[gradIdx], featureValue));
                }
            }
        }

        if (_categoricalEmbeddings != null && _categoricalEmbeddingsGrad != null && _categoricalIndicesCache != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < NumCategoricalFeatures; f++)
                {
                    int catIdx = _categoricalIndicesCache[b, f];
                    int featureIdx = NumNumericalFeatures + f;
                    for (int d = 0; d < embeddingDim; d++)
                    {
                        int gradIdx = (b * TotalFeatures + featureIdx) * embeddingDim + d;
                        int embIdx = catIdx * embeddingDim + d;
                        _categoricalEmbeddingsGrad[f][embIdx] = NumOps.Add(
                            _categoricalEmbeddingsGrad[f][embIdx],
                            gradEmbedded[gradIdx]);
                    }
                }
            }
        }
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
        return _interactingLayers[^1].GetAttentionScores();
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Scale learning rate by batch size to normalize gradients that were accumulated across samples
        int batchSize = _numericalFeaturesCache?.Shape[0] ?? 1;
        var scaledLr = batchSize > 1
            ? NumOps.Divide(learningRate, NumOps.FromDouble(batchSize))
            : learningRate;

        for (int i = 0; i < _numericalEmbeddings.Length; i++)
        {
            _numericalEmbeddings[i] = NumOps.Subtract(
                _numericalEmbeddings[i],
                NumOps.Multiply(scaledLr, _numericalEmbeddingsGrad[i]));
        }

        if (_categoricalEmbeddings != null && _categoricalEmbeddingsGrad != null)
        {
            for (int f = 0; f < _categoricalEmbeddings.Length; f++)
            {
                var emb = _categoricalEmbeddings[f];
                var grad = _categoricalEmbeddingsGrad[f];
                for (int i = 0; i < emb.Length; i++)
                {
                    emb[i] = NumOps.Subtract(
                        emb[i],
                        NumOps.Multiply(scaledLr, grad[i]));
                }
            }
        }

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
        _numericalFeaturesCache = null;
        _categoricalIndicesCache = null;
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

}

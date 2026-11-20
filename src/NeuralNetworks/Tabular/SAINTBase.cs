using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
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
/// Base class for SAINT (Self-Attention and Intersample Attention Transformer).
/// </summary>
/// <remarks>
/// <para>
/// SAINT applies two types of attention in alternating layers:
/// 1. Column attention: Self-attention over features (like FT-Transformer)
/// 2. Row attention: Inter-sample attention comparing samples in a batch
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of SAINT as looking at your data from two perspectives:
///
/// - **Column attention**: "Which features are related to each other?"
///   (e.g., income and education level might be correlated)
/// - **Row attention**: "Which samples in my batch are similar?"
///   (e.g., customers with similar profiles might have similar behavior)
///
/// By combining both views, SAINT can learn patterns that other models miss.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class SAINTBase<T> : IParameterSource<T>
{
    protected readonly SAINTOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalEmbeddedFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected IEngine Engine => AiDotNetEngine.Current;
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    private readonly FullyConnectedLayer<T> _numericalEmbedding;
    private readonly Tensor<T>[]? _categoricalEmbeddings;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T>? _columnEmbeddings;

    // Transformer layers (alternating column and row attention)
    private readonly List<MultiHeadAttentionLayer<T>> _columnAttentionLayers;
    private readonly List<MultiHeadAttentionLayer<T>>? _rowAttentionLayers;
    private readonly List<FullyConnectedLayer<T>> _ffnLayers;
    private readonly List<LayerNormalizationLayer<T>>? _layerNorms;

    // MLP head
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// Extra trainable layers a subclass contributes, folded after the shared backbone.
    /// </summary>
    protected virtual IEnumerable<IParameterSource<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(SAINTBase<T>));

    /// <summary>The single ordered traversal of this model's parameter-bearing components.</summary>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            // Every attention projection is sized entirely by EmbeddingDimension and NumHeads.
            // Materialize on first parameter-surface access so a fresh SAINT checkpoint describes
            // the same model as one that has already run a prediction.
            foreach (var layer in _columnAttentionLayers)
                layer.MaterializeParameters();
            if (_rowAttentionLayers != null)
            {
                foreach (var layer in _rowAttentionLayers)
                    layer.MaterializeParameters();
            }

            var registry = new ParameterComponentRegistry<T>();
            registry.Register("00000000/numerical", _numericalEmbedding);
            registry.Register("00000001/categorical",
                new TensorCollectionParameterSource<T>(() => _categoricalEmbeddings));
            registry.Register("00000002/column",
                new TensorFieldParameterSource<T>(() => _columnEmbeddings));

            for (int i = 0; i < _columnAttentionLayers.Count; i++)
                registry.Register($"00000003/{i:D8}", _columnAttentionLayers[i]);

            if (_rowAttentionLayers != null)
                for (int i = 0; i < _rowAttentionLayers.Count; i++)
                    registry.Register($"00000004/{i:D8}", _rowAttentionLayers[i]);

            for (int i = 0; i < _ffnLayers.Count; i++)
                registry.Register($"00000005/{i:D8}", _ffnLayers[i]);

            if (_layerNorms != null)
                for (int i = 0; i < _layerNorms.Count; i++)
                    registry.Register($"00000006/{i:D8}", _layerNorms[i]);

            for (int i = 0; i < _mlpLayers.Count; i++)
                registry.Register($"00000007/{i:D8}", _mlpLayers[i]);

            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc />
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <inheritdoc />
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <inheritdoc />
    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

    /// <summary>
    /// Initializes a new instance of the SAINTBase class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected SAINTBase(int numNumericalFeatures, SAINTOptions<T>? options = null)
    {
        Options = options ?? new SAINTOptions<T>();
        Options.Validate();
        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        TotalEmbeddedFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        if (TotalEmbeddedFeatures == 0)
        {
            throw new ArgumentException("Model must have at least one feature (numerical or categorical)");
        }

        // Numerical feature embedding
        _numericalEmbedding = Options.HiddenVectorActivation is null
            ? new FullyConnectedLayer<T>(
                1,
                Options.EmbeddingDimension,
                Options.HiddenActivation ?? new GELUActivation<T>())
            : FullyConnectedLayer<T>.WithVectorActivation(
                1,
                Options.EmbeddingDimension,
                Options.HiddenVectorActivation);

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

        // Column embeddings (positional encoding for features)
        _columnEmbeddings = new Tensor<T>(new[] { TotalEmbeddedFeatures, Options.EmbeddingDimension });
        InitializeEmbedding(_columnEmbeddings);

        // Initialize transformer layers
        _columnAttentionLayers = [];
        _ffnLayers = [];

        if (Options.UseIntersampleAttention)
        {
            _rowAttentionLayers = [];
        }

        if (Options.UseLayerNorm)
        {
            _layerNorms = [];
        }

        for (int i = 0; i < Options.NumLayers; i++)
        {
            // Column attention layer
            _columnAttentionLayers.Add(new MultiHeadAttentionLayer<T>(
                Options.NumHeads,
                Options.EmbeddingDimension / Options.NumHeads));

            // Row attention layer (if enabled)
            if (Options.UseIntersampleAttention)
            {
                (_rowAttentionLayers ?? throw new InvalidOperationException("Row attention layers not initialized.")).Add(
                    new MultiHeadAttentionLayer<T>(
                        Options.NumHeads,
                        Options.EmbeddingDimension / Options.NumHeads));
            }

            // Feed-forward network
            _ffnLayers.Add(Options.HiddenVectorActivation is null
                ? new FullyConnectedLayer<T>(
                    Options.EmbeddingDimension,
                    Options.EmbeddingDimension,
                    Options.HiddenActivation ?? new GELUActivation<T>())
                : FullyConnectedLayer<T>.WithVectorActivation(
                    Options.EmbeddingDimension,
                    Options.EmbeddingDimension,
                    Options.HiddenVectorActivation));

            // Layer normalizations
            if (Options.UseLayerNorm)
            {
                var layerNorms = _layerNorms ?? throw new InvalidOperationException("Layer norms not initialized.");
                layerNorms.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                layerNorms.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                if (Options.UseIntersampleAttention)
                {
                    layerNorms.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                }
            }
        }

        // MLP head
        _mlpLayers = [];
        int mlpInput = TotalEmbeddedFeatures * Options.EmbeddingDimension;
        foreach (var hiddenDim in Options.MLPHiddenDimensions)
        {
            _mlpLayers.Add(Options.HiddenVectorActivation is null
                ? new FullyConnectedLayer<T>(
                    mlpInput,
                    hiddenDim,
                    Options.HiddenActivation ?? new GELUActivation<T>())
                : FullyConnectedLayer<T>.WithVectorActivation(
                    mlpInput,
                    hiddenDim,
                    Options.HiddenVectorActivation));
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
    /// Embeds input features into the transformer space.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var colEmbeddings = _columnEmbeddings
            ?? throw new InvalidOperationException("Column embeddings not initialized.");

        int batchSize = numericalFeatures.Shape[0];
        int embeddingDim = Options.EmbeddingDimension;
        var columns = colEmbeddings.Reshape(1, TotalEmbeddedFeatures, embeddingDim);

        // Embed numerical features: every scalar goes through the shared 1 -> embedding_dim projection as its own row
        var values = TabularMetrics.Flat(numericalFeatures, batchSize * NumNumericalFeatures)
            .Reshape(batchSize * NumNumericalFeatures, 1);
        var numerical = Engine.TensorAdd(
            _numericalEmbedding.Forward(values).Reshape(batchSize, NumNumericalFeatures, embeddingDim),
            Engine.TensorNarrow(columns, 1, 0, NumNumericalFeatures));

        if (NumCategoricalFeatures == 0)
        {
            return numerical;
        }

        // Embed categorical features (left at zero when no indices are given)
        var parts = new Tensor<T>[NumCategoricalFeatures + 1];
        parts[0] = numerical;
        var categoricalEmbeddings = _categoricalEmbeddings;
        for (int f = 0; f < NumCategoricalFeatures; f++)
        {
            int featureIdx = NumNumericalFeatures + f;
            parts[f + 1] = categoricalIndices != null && categoricalEmbeddings != null
                ? Engine.TensorAdd(
                    Engine.TensorIndexSelect(
                        categoricalEmbeddings[f],
                        Tensor<int>.FromVector(categoricalIndices.GetColumn(f).GetSubVector(0, batchSize)),
                        0).Reshape(batchSize, 1, embeddingDim),
                    Engine.TensorNarrow(columns, 1, featureIdx, 1))
                : new Tensor<T>(new[] { batchSize, 1, embeddingDim });
        }

        return Engine.TensorConcatenate(parts, 1);
    }

    /// <summary>
    /// Performs the forward pass through the SAINT backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);

        var current = embedded;

        // Process through transformer layers
        // Validate and hoist layer norms upfront to avoid repeated null checks inside the loop
        var norms = Options.UseLayerNorm
            ? _layerNorms ?? throw new InvalidOperationException("Layer norms not initialized when UseLayerNorm is true.")
            : (IReadOnlyList<LayerNormalizationLayer<T>>)Array.Empty<LayerNormalizationLayer<T>>();
        int normIdx = 0;
        for (int layer = 0; layer < Options.NumLayers; layer++)
        {
            // Column attention (self-attention over features)
            var colAttnInput = current;
            if (Options.UsePreNorm && Options.UseLayerNorm)
            {
                colAttnInput = ApplyLayerNorm(norms[normIdx++], colAttnInput, batchSize);
            }

            var colAttnOutput = ApplyColumnAttention(_columnAttentionLayers[layer], colAttnInput, batchSize);

            // Residual connection
            current = AddTensors(current, colAttnOutput);

            if (!Options.UsePreNorm && Options.UseLayerNorm)
            {
                current = ApplyLayerNorm(norms[normIdx++], current, batchSize);
            }

            // Row attention (inter-sample attention, if enabled)
            if (Options.UseIntersampleAttention && _rowAttentionLayers != null)
            {
                var rowAttnInput = current;
                if (Options.UsePreNorm && Options.UseLayerNorm)
                {
                    rowAttnInput = ApplyLayerNorm(norms[normIdx++], rowAttnInput, batchSize);
                }

                var rowAttnOutput = ApplyRowAttention(_rowAttentionLayers[layer], rowAttnInput, batchSize);

                // Residual connection
                current = AddTensors(current, rowAttnOutput);

                if (!Options.UsePreNorm && Options.UseLayerNorm)
                {
                    current = ApplyLayerNorm(norms[normIdx++], current, batchSize);
                }
            }

            // Feed-forward network
            var ffnInput = current;
            if (Options.UsePreNorm && Options.UseLayerNorm)
            {
                ffnInput = ApplyLayerNorm(norms[normIdx++], ffnInput, batchSize);
            }

            var ffnOutput = ApplyFeedForward(_ffnLayers[layer], ffnInput, batchSize);

            // Residual connection
            current = AddTensors(current, ffnOutput);

            if (!Options.UsePreNorm && Options.UseLayerNorm)
            {
                current = ApplyLayerNorm(norms[normIdx++], current, batchSize);
            }
        }

        // Flatten and apply MLP head
        var flattened = TabularMetrics.Flat(current, batchSize * TotalEmbeddedFeatures * Options.EmbeddingDimension)
            .Reshape(batchSize, TotalEmbeddedFeatures * Options.EmbeddingDimension);

        var mlpOutput = flattened;
        foreach (var mlpLayer in _mlpLayers)
        {
            mlpOutput = mlpLayer.Forward(mlpOutput);
        }

        return mlpOutput;
    }

    private Tensor<T> ApplyColumnAttention(MultiHeadAttentionLayer<T> layer, Tensor<T> input, int batchSize)
    {
        // Column attention: attention over the feature dimension
        // Input shape: [batch, features, embedding_dim]
        // Each sample attends over its features independently: one batched [batch, features, dim] call
        var samples = input.Reshape(batchSize, TotalEmbeddedFeatures, Options.EmbeddingDimension);
        return layer.Forward(samples).Reshape(input._shape);
    }

    private Tensor<T> ApplyRowAttention(MultiHeadAttentionLayer<T> layer, Tensor<T> input, int batchSize)
    {
        // Row attention: attention over the batch dimension for each feature
        // Input shape: [batch, features, embedding_dim]
        // Each feature attends across samples: transpose to [features, batch, dim] and attend in one call
        var byFeature = Engine.TensorPermute(
            input.Reshape(batchSize, TotalEmbeddedFeatures, Options.EmbeddingDimension), new[] { 1, 0, 2 });
        var attended = layer.Forward(byFeature);
        return Engine.TensorPermute(attended, new[] { 1, 0, 2 }).Reshape(input._shape);
    }

    private Tensor<T> ApplyFeedForward(FullyConnectedLayer<T> layer, Tensor<T> input, int batchSize)
    {
        // The feed-forward network acts on every (sample, feature) embedding independently
        var rows = input.Reshape(batchSize * TotalEmbeddedFeatures, Options.EmbeddingDimension);
        return layer.Forward(rows).Reshape(input._shape);
    }

    private Tensor<T> ApplyLayerNorm(LayerNormalizationLayer<T> norm, Tensor<T> input, int batchSize)
    {
        // Layer norm over the embedding dimension of every (sample, feature) embedding
        var rows = input.Reshape(batchSize * TotalEmbeddedFeatures, Options.EmbeddingDimension);
        return norm.Forward(rows).Reshape(input._shape);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        _numericalEmbedding.UpdateParameters(learningRate);

        foreach (var layer in _columnAttentionLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        if (_rowAttentionLayers != null)
        {
            foreach (var layer in _rowAttentionLayers)
            {
                layer.UpdateParameters(learningRate);
            }
        }

        foreach (var layer in _ffnLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// The live tensors a tape training step updates: the embeddings, the column and row attention, feed-forward and norm layers, the MLP and the task head.
    /// </summary>
    protected IEnumerable<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>>();
        if (_categoricalEmbeddings != null) tensors.AddRange(_categoricalEmbeddings);
        if (_columnEmbeddings != null) tensors.Add(_columnEmbeddings);
        var layers = new List<ILayer<T>> { _numericalEmbedding };
        layers.AddRange(_columnAttentionLayers);
        if (_rowAttentionLayers != null) layers.AddRange(_rowAttentionLayers);
        layers.AddRange(_ffnLayers);
        if (_layerNorms != null) layers.AddRange(_layerNorms);
        layers.AddRange(_mlpLayers);
        layers.AddRange(GetExtraTrainableLayers().OfType<ILayer<T>>());
        tensors.AddRange(TabularTraining.LayerTensors(layers));
        return tensors;
    }

    /// <summary>
    /// One tape training step: gradients of <paramref name="objective"/> over every trainable tensor, then SGD.
    /// </summary>
    /// <returns>The loss before the update.</returns>
    protected T TrainWithTape(Func<Tensor<T>> objective, T learningRate)
    {
        try
        {
            return TabularTraining.SgdStep(GetTrainableTensors, objective, learningRate);
        }
        finally
        {
            ResetState();
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _numericalEmbedding.ResetState();

        foreach (var layer in _columnAttentionLayers)
        {
            layer.ResetState();
        }

        if (_rowAttentionLayers != null)
        {
            foreach (var layer in _rowAttentionLayers)
            {
                layer.ResetState();
            }
        }

        foreach (var layer in _ffnLayers)
        {
            layer.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }
}

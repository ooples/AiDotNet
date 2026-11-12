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
using AiDotNet.Training;

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
public abstract partial class AutoIntBase<T> : IParameterSource<T>
{
    protected readonly AutoIntOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected IEngine Engine => AiDotNetEngine.Current;
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _numericalEmbeddings;
    private readonly Tensor<T>[]? _categoricalEmbeddings;

    // Interacting layers (multi-head self-attention)
    private readonly List<InteractingLayer<T>> _interactingLayers;

    // MLP output
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    // Caches for the embedding-gradient backward
    [Scratch]
    private Tensor<T>? _numericalFeaturesCache;
    [Scratch]
    private Matrix<int>? _categoricalIndicesCache;

    // Batch size of the last forward, which scales the learning rate in UpdateParameters. Kept apart
    // from the caches above so that skipping them when no backward runs cannot change the update.
    private int _forwardBatchSize;

    // Embedding gradients
    private readonly Tensor<T> _numericalEmbeddingsGrad;
    private readonly Tensor<T>[]? _categoricalEmbeddingsGrad;

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
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(AutoIntBase<T>));

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
            registry.Register("00000000/numerical",
                new TensorFieldParameterSource<T>(() => _numericalEmbeddings));
            registry.Register("00000001/categorical",
                new TensorCollectionParameterSource<T>(() => _categoricalEmbeddings));

            for (int i = 0; i < _interactingLayers.Count; i++)
                registry.Register($"00000002/{i:D8}", _interactingLayers[i]);

            for (int i = 0; i < _mlpLayers.Count; i++)
                registry.Register($"00000003/{i:D8}", _mlpLayers[i]);
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
    /// Reads every parameter in traversal order. <see cref="SetParameters"/> reads it back in the
    /// same order, and <see cref="ParameterCount"/> is the length of what this returns.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

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
            _mlpLayers.Add(Options.HiddenVectorActivation is null
                ? new FullyConnectedLayer<T>(
                    mlpInput,
                    hiddenDim,
                    Options.HiddenActivation ?? new ReLUActivation<T>())
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
        // N(0, EmbeddingInitScale^2) in one engine call, seeded from this model's (secure) Random.
        Engine.TensorNormalInPlace(embedding, 0.0, Options.EmbeddingInitScale, _random.Next());
    }

    /// <summary>Computes the scalar objective's gradients and applies one SGD step to the whole model.</summary>
    /// <param name="objective">Computes the batch-averaged scalar loss through differentiable engine operations.</param>
    /// <param name="learningRate">Step size applied to the computed gradients.</param>
    /// <returns>The loss before updating the model.</returns>
    protected T TrainWithTape(Func<Tensor<T>> objective, T learningRate)
    {
        var parameters = new List<Tensor<T>> { _numericalEmbeddings };
        if (_categoricalEmbeddings != null) parameters.AddRange(_categoricalEmbeddings);
        foreach (var layer in _interactingLayers) parameters.AddRange(layer.GetTrainableParameters());
        foreach (var layer in _mlpLayers) parameters.AddRange(layer.GetTrainableParameters());
        foreach (var layer in GetExtraTrainableLayers().OfType<ITrainableLayer<T>>())
            parameters.AddRange(layer.GetTrainableParameters());
        try
        {
            // The objective already averages over the batch; do not divide its gradients again.
            return TapeTrainingStepper<T>.EagerSgdObjectiveStep(() => parameters, objective, learningRate);
        }
        finally
        {
            foreach (var parameter in parameters.Distinct())
                Engine.InvalidatePersistentTensor(parameter);
            ResetState();
        }
    }

    /// <summary>
    /// Embeds input features.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        if (numericalFeatures.Rank != 2 || numericalFeatures.Shape[1] != NumNumericalFeatures || batchSize == 0)
            throw new ArgumentException("Expected a nonempty [batch, numerical features] tensor.", nameof(numericalFeatures));
        var parts = new List<Tensor<T>>();
        if (NumNumericalFeatures > 0)
        {
            var numerical = Engine.Reshape(numericalFeatures, new[] { batchSize, NumNumericalFeatures, 1 });
            var table = Engine.Reshape(_numericalEmbeddings, new[] { 1, NumNumericalFeatures, Options.EmbeddingDimension });
            parts.Add(Engine.TensorMultiply(numerical, table));
        }

        if (categoricalIndices != null && _categoricalEmbeddings != null)
        {
            if (categoricalIndices.Rows != batchSize || categoricalIndices.Columns != NumCategoricalFeatures)
                throw new ArgumentException("Categorical indices must match the batch and categorical feature counts.", nameof(categoricalIndices));
            var cardinalities = Options.CategoricalCardinalities
                ?? throw new InvalidOperationException("AutoIntBase: CategoricalCardinalities not configured.");

            // [feature, batch] index table, validated in one pass: the first out-of-range entry in
            // feature-major order is the one reported, as before.
            var indexRows = Tensor<int>.FromMatrix(categoricalIndices.Transpose());
            var limit = Engine.TensorBroadcastTo(
                Tensor<int>.FromVector(new Vector<int>(cardinalities).GetSubVector(0, NumCategoricalFeatures), new[] { NumCategoricalFeatures, 1 }),
                new[] { NumCategoricalFeatures, batchSize });
            var invalid = Engine.TensorAdd(Engine.TensorLessThan(indexRows, 0), Engine.TensorGreaterEqual(indexRows, limit));
            if (Engine.TensorMaxValue(invalid) > 0)
            {
                int first = Engine.TensorArgMax(Engine.Reshape(invalid, new[] { NumCategoricalFeatures * batchSize }), 0).ToArray()[0];
                int badFeature = first / batchSize;
                int badRow = first % batchSize;
                throw new ArgumentOutOfRangeException(nameof(categoricalIndices),
                    $"Categorical index {categoricalIndices[badRow, badFeature]} for feature {badFeature} (batch {badRow}) is out of range [0, {cardinalities[badFeature]}).");
            }

            for (int f = 0; f < NumCategoricalFeatures; f++)
            {
                var indices = Engine.Reshape(Engine.TensorSlice(indexRows, new[] { f, 0 }, new[] { 1, batchSize }), new[] { batchSize });
                var selected = Engine.TensorEmbeddingLookup<T, int>(_categoricalEmbeddings[f], indices);
                parts.Add(Engine.Reshape(selected, new[] { batchSize, 1, Options.EmbeddingDimension }));
            }
        }
        else if (NumCategoricalFeatures > 0)
        {
            parts.Add(new Tensor<T>(new[] { batchSize, NumCategoricalFeatures, Options.EmbeddingDimension }));
        }
        return parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), axis: 1);
    }

    /// <summary>
    /// Performs the forward pass through the AutoInt backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        _forwardBatchSize = batchSize;
        ManualBackwardCache<T>.SaveForBackward(ref _numericalFeaturesCache, numericalFeatures);
        if (categoricalIndices is not null)
            ManualBackwardCache<T>.SaveForBackward(ref _categoricalIndicesCache, categoricalIndices);
        else
            _categoricalIndicesCache = null;

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);

        // Apply interacting layers
        var current = embedded;
        foreach (var layer in _interactingLayers)
        {
            current = layer.Forward(current);
        }
        // Reshape through the engine so the tape can carry gradients back to the embeddings.
        var flattened = Engine.Reshape(current, new[] { batchSize, TotalFeatures * Options.EmbeddingDimension });
        // MLP
        var mlpOutput = flattened;
        foreach (var mlpLayer in _mlpLayers)
        {
            mlpOutput = mlpLayer.Forward(mlpOutput);
        }

        return mlpOutput;
    }

    private void AccumulateEmbeddingGradients(Tensor<T> gradEmbedded)
    {
        // Reset gradients
        Engine.TensorFill(_numericalEmbeddingsGrad, NumOps.Zero);

        if (_categoricalEmbeddingsGrad != null)
        {
            foreach (var grad in _categoricalEmbeddingsGrad)
            {
                Engine.TensorFill(grad, NumOps.Zero);
            }
        }

        if (_numericalFeaturesCache == null)
        {
            return;
        }

        int batchSize = gradEmbedded.Shape[0];
        int embeddingDim = Options.EmbeddingDimension;
        var grad3 = Engine.Reshape(gradEmbedded, new[] { batchSize, TotalFeatures, embeddingDim });

        // Numerical embeddings: dE[f, :] = sum_b grad[b, f, :] * x[b, f]
        if (NumNumericalFeatures > 0)
        {
            var numericalGrad = Engine.TensorSlice(grad3, new[] { 0, 0, 0 }, new[] { batchSize, NumNumericalFeatures, embeddingDim });
            var x = Engine.TensorBroadcastTo(
                Engine.Reshape(_numericalFeaturesCache, new[] { batchSize, NumNumericalFeatures, 1 }),
                new[] { batchSize, NumNumericalFeatures, embeddingDim });
            Engine.TensorCopy(Engine.ReduceSum(Engine.TensorMultiply(numericalGrad, x), new[] { 0 }, false), _numericalEmbeddingsGrad);
        }

        if (_categoricalEmbeddings != null && _categoricalEmbeddingsGrad != null && _categoricalIndicesCache != null)
        {
            // Categorical embeddings: scatter-add each row's gradient into its looked-up table row.
            var indexRows = Tensor<int>.FromMatrix(_categoricalIndicesCache.Transpose());
            for (int f = 0; f < NumCategoricalFeatures; f++)
            {
                var rows = Engine.Reshape(
                    Engine.TensorSlice(grad3, new[] { 0, NumNumericalFeatures + f, 0 }, new[] { batchSize, 1, embeddingDim }),
                    new[] { batchSize, embeddingDim });
                var index = Engine.Reshape(Engine.TensorSlice(indexRows, new[] { f, 0 }, new[] { 1, batchSize }), new[] { batchSize });
                Engine.TensorCopy(Engine.TensorScatterAdd(_categoricalEmbeddingsGrad[f], index, rows, 0), _categoricalEmbeddingsGrad[f]);
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
        int batchSize = _forwardBatchSize > 0 ? _forwardBatchSize : 1;
        var scaledLr = batchSize > 1
            ? NumOps.Divide(learningRate, NumOps.FromDouble(batchSize))
            : learningRate;

        // Note: _numericalEmbeddings is readonly, so we update in-place with subtract
        Engine.TensorSubtractInPlace(_numericalEmbeddings, Engine.TensorMultiplyScalar(_numericalEmbeddingsGrad, scaledLr));

        if (_categoricalEmbeddings != null && _categoricalEmbeddingsGrad != null)
        {
            for (int f = 0; f < _categoricalEmbeddings.Length; f++)
            {
                Engine.TensorSubtractInPlace(_categoricalEmbeddings[f],
                    Engine.TensorMultiplyScalar(_categoricalEmbeddingsGrad[f], scaledLr));
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
        _forwardBatchSize = 0;

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

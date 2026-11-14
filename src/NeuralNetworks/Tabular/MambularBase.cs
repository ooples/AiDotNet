using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using System;
using System.Collections.Generic;
using AiDotNet.Models.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using System.Linq;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for Mambular (State Space Models for Tabular Data).
/// </summary>
/// <remarks>
/// <para>
/// Mambular applies the Mamba architecture to tabular data:
/// 1. Features are embedded and treated as a sequence
/// 2. A stack of pre-norm residual Mamba layers, x + Mamba(RMSNorm(x)), processes the feature sequence. Each
///    Mamba layer is the full selective state space block of Gu and Dao (2023): input projection, causal
///    depthwise conv1d, SiLU, the selective S6 scan with input-dependent delta, B and C, SiLU output gating
///    and the output projection (<see cref="MambaBlock{T}"/>).
/// 3. Final representation is used for prediction
/// </para>
/// <para>
/// <b>For Beginners:</b> Mambular is an alternative to transformers that:
///
/// - **Scales linearly**: O(n) instead of O(n²) with sequence length
/// - **Has memory**: Can remember information across the feature sequence
/// - **Is selective**: Learns what to remember and what to forget
///
/// This makes it efficient for tabular data with many features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class MambularBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    protected readonly MambularOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _numericalEmbeddings;
    private readonly Tensor<T>[]? _categoricalEmbeddings;

    // Mamba layers
    private readonly List<MambaBlock<T>> _mambaBlocks;

    // Pre-norm of each residual Mamba layer (Mambular: x + Mamba(RMSNorm(x)))
    private readonly List<RMSNormalizationLayer<T>> _blockNorms;

    // MLP head
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

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
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(MambularBase<T>));

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

            for (int i = 0; i < _mambaBlocks.Count; i++)
                registry.Register($"00000002/{i:D8}", _mambaBlocks[i]);

            for (int i = 0; i < _blockNorms.Count; i++)
                registry.Register($"00000004/{i:D8}", _blockNorms[i]);

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
    /// Initializes a new instance of the MambularBase class.
    /// </summary>
    protected MambularBase(int numNumericalFeatures, MambularOptions<T>? options = null)
    {
        Options = options ?? new MambularOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        TotalFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        if (TotalFeatures == 0)
        {
            throw new ArgumentException("Model must have at least one feature");
        }

        // Feature embeddings
        _numericalEmbeddings = new Tensor<T>(new[] { NumNumericalFeatures, Options.EmbeddingDimension });
        InitializeWeights(_numericalEmbeddings);

        if (NumCategoricalFeatures > 0 && Options.CategoricalCardinalities != null)
        {
            _categoricalEmbeddings = new Tensor<T>[NumCategoricalFeatures];
            for (int i = 0; i < NumCategoricalFeatures; i++)
            {
                int cardinality = Options.CategoricalCardinalities[i];
                _categoricalEmbeddings[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
                InitializeWeights(_categoricalEmbeddings[i]);
            }
        }

        // Mamba blocks
        _mambaBlocks = [];
        _blockNorms = [];
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _blockNorms.Add(new RMSNormalizationLayer<T>(Options.EmbeddingDimension));
            _mambaBlocks.Add(new MambaBlock<T>(
                sequenceLength: TotalFeatures,
                modelDimension: Options.EmbeddingDimension,
                stateDimension: Options.StateDimension,
                expandFactor: Options.ExpansionFactor,
                convKernelSize: Options.ConvKernelSize,
                deltaMin: Options.DeltaMin,
                deltaMax: Options.DeltaMax));
        }

        // MLP head
        _mlpLayers = [];
        int mlpInput = Options.UseBidirectional
            ? TotalFeatures * Options.EmbeddingDimension * 2
            : TotalFeatures * Options.EmbeddingDimension;

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

    private void InitializeWeights(Tensor<T> tensor)
    {
        var scale = NumOps.FromDouble(Options.InitScale);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(_random.NextGaussian()), scale);
        }
    }

    /// <summary>
    /// Embeds input features.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        int embeddingDim = Options.EmbeddingDimension;

        // Numerical features: embedded[b, f, :] = x[b, f] * E_num[f, :]
        var values = TabularMetrics.Flat(numericalFeatures, batchSize * NumNumericalFeatures)
            .Reshape(batchSize, NumNumericalFeatures, 1);
        var numerical = Engine.TensorMultiply(
            values, _numericalEmbeddings.Reshape(1, NumNumericalFeatures, embeddingDim));

        if (NumCategoricalFeatures == 0)
        {
            return numerical;
        }

        // Categorical features: embedded[b, NumNumerical + f, :] = E_cat_f[index[b, f], :], zeros without indices
        var parts = new Tensor<T>[NumCategoricalFeatures + 1];
        parts[0] = numerical;
        var categoricalEmbeddings = _categoricalEmbeddings;
        for (int f = 0; f < NumCategoricalFeatures; f++)
        {
            parts[f + 1] = categoricalIndices != null && categoricalEmbeddings != null
                ? Engine.TensorIndexSelect(
                    categoricalEmbeddings[f],
                    Tensor<int>.FromVector(categoricalIndices.GetColumn(f).GetSubVector(0, batchSize)),
                    0).Reshape(batchSize, 1, embeddingDim)
                : new Tensor<T>(new[] { batchSize, 1, embeddingDim });
        }

        return Engine.TensorConcatenate(parts, 1);
    }

    /// <summary>
    /// Performs the forward pass through the Mambular backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);

        // Forward direction
        var forward = ApplyMambaLayers(embedded);

        Tensor<T> combined;
        if (Options.UseBidirectional)
        {
            // Backward direction (reverse sequence)
            var reversed = ReverseSequence(embedded, batchSize, TotalFeatures, Options.EmbeddingDimension);
            var backward = ApplyMambaLayers(reversed);
            backward = ReverseSequence(backward, batchSize, TotalFeatures, Options.EmbeddingDimension);

            // Concatenate forward and backward
            int flatWidth = TotalFeatures * Options.EmbeddingDimension;
            combined = Engine.TensorConcatenate(
                new[] { forward.Reshape(batchSize, flatWidth), backward.Reshape(batchSize, flatWidth) }, 1);
        }
        else
        {
            // Flatten forward output
            combined = forward.Reshape(batchSize, TotalFeatures * Options.EmbeddingDimension);
        }

        // MLP head
        var mlpOutput = combined;
        foreach (var layer in _mlpLayers)
        {
            mlpOutput = layer.Forward(mlpOutput);
        }

        return mlpOutput;
    }

    /// <summary>
    /// The residual Mamba stack over a [batch, features, embedding] sequence: x = x + Mamba(RMSNorm(x)) per layer.
    /// </summary>
    private Tensor<T> ApplyMambaLayers(Tensor<T> sequence)
    {
        var current = sequence;
        for (int i = 0; i < _mambaBlocks.Count; i++)
        {
            var normed = _blockNorms[i].Forward(current);
            current = Engine.TensorAdd(current, _mambaBlocks[i].Forward(normed));
        }

        return current;
    }

    /// <summary>
    /// The live tensors a tape training step updates: the embeddings, every Mamba layer and norm, the MLP and
    /// the task head a subclass adds.
    /// </summary>
    protected IEnumerable<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>> { _numericalEmbeddings };
        if (_categoricalEmbeddings != null) tensors.AddRange(_categoricalEmbeddings);

        var layers = new List<ILayer<T>>();
        layers.AddRange(_blockNorms);
        layers.AddRange(_mambaBlocks);
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

    private Tensor<T> ReverseSequence(Tensor<T> input, int batchSize, int seqLen, int dim)
    {
        return Engine.TensorFlip(input.Reshape(batchSize, seqLen, dim), new[] { 1 }).Reshape(input._shape);
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        foreach (var norm in _blockNorms)
        {
            norm.UpdateParameters(learningRate);
        }

        foreach (var block in _mambaBlocks)
        {
            block.UpdateParameters(learningRate);
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
        foreach (var norm in _blockNorms)
        {
            norm.ResetState();
        }

        foreach (var block in _mambaBlocks)
        {
            block.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }
}

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

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for Mambular (State Space Models for Tabular Data).
/// </summary>
/// <remarks>
/// <para>
/// Mambular applies the Mamba architecture to tabular data:
/// 1. Features are embedded and treated as a sequence
/// 2. Selective State Space Model (S4/Mamba) processes the sequence
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
    private readonly List<MambaBlock> _mambaBlocks;

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
        for (int i = 0; i < Options.NumLayers; i++)
        {
            _mambaBlocks.Add(new MambaBlock(
                Options.EmbeddingDimension,
                Options.StateDimension,
                Options.InnerDimension,
                Options.ConvKernelSize,
                Options.DeltaMin,
                Options.DeltaMax,
                _random));
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
        var forward = embedded;
        foreach (var block in _mambaBlocks)
        {
            forward = block.Forward(forward, batchSize, TotalFeatures, Options.EmbeddingDimension);
        }

        Tensor<T> combined;
        if (Options.UseBidirectional)
        {
            // Backward direction (reverse sequence)
            var reversed = ReverseSequence(embedded, batchSize, TotalFeatures, Options.EmbeddingDimension);
            var backward = reversed;
            foreach (var block in _mambaBlocks)
            {
                backward = block.Forward(backward, batchSize, TotalFeatures, Options.EmbeddingDimension);
            }
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

    private Tensor<T> ReverseSequence(Tensor<T> input, int batchSize, int seqLen, int dim)
    {
        return Engine.TensorFlip(input.Reshape(batchSize, seqLen, dim), new[] { 1 }).Reshape(input._shape);
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
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
        foreach (var block in _mambaBlocks)
        {
            block.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Simplified Mamba block for tabular data.
    /// </summary>
    private class MambaBlock : IParameterSource<T>
    {
        private readonly int _modelDim;
        private readonly int _stateDim;
        private readonly int _innerDim;

        // SSM parameters
        private readonly Tensor<T> _A;  // State transition
        private readonly Tensor<T> _B;  // Input projection
        private readonly Tensor<T> _C;  // Output projection
        private readonly Tensor<T> _D;  // Skip connection

        // Projections
        private readonly Tensor<T> _inProj;
        private readonly Tensor<T> _outProj;

        // Convolution
        private readonly Tensor<T> _convWeight;

        // Delta (discretization). Built once in the constructor and only ever read afterwards, and
        // it is yielded from Tensors() below, so it travels in the parameter vector exactly like the
        // seven weights above it. It was briefly labelled Scratch, which would have declared the
        // opposite -- a value the restore path is free to discard -- while the vector was still
        // carrying it. A field cannot be both.
        private readonly Tensor<T> _deltaProj;

        /// <summary>
        /// The block's tensors in serialization order. Count, read and restore all walk THIS, so a
        /// tensor added later joins all three at once instead of only the sum.
        /// </summary>
        private IEnumerable<Tensor<T>> Tensors()
        {
            yield return _A;
            yield return _B;
            yield return _C;
            yield return _D;
            yield return _inProj;
            yield return _outProj;
            yield return _convWeight;
            yield return _deltaProj;
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
        public Vector<T> GetParameters()
        {
            var result = new Vector<T>(checked((int)ParameterCount));
            var destination = result.AsWritableSpan();
            int offset = 0;

            foreach (var tensor in Tensors())
            {
                tensor.AsSpan().CopyTo(destination.Slice(offset, tensor.Length));
                offset += tensor.Length;
            }

            return result;
        }

        /// <summary>Writes THROUGH the tensors, so the block computes with what was restored.</summary>
        /// <remarks>
        /// Rebinding the fields instead would leave any view or gradient buffer sharing the old
        /// buffer, which is the stale-weights defect: the restore reports success and the block goes
        /// on computing with its pre-restore values.
        /// </remarks>
        public void SetParameters(Vector<T> parameters)
        {
            if (parameters is null) throw new ArgumentNullException(nameof(parameters));

            long expected = ParameterCount;
            if (parameters.Length != expected)
            {
                throw new ArgumentException(
                    $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
            }

            int offset = 0;
            var source = parameters.AsSpan();
            foreach (var tensor in Tensors())
            {
                source.Slice(offset, tensor.Length).CopyTo(tensor.AsWritableSpan());
                offset += tensor.Length;
            }
        }

        public MambaBlock(
            int modelDim,
            int stateDim,
            int innerDim,
            int convKernelSize,
            double deltaMin,
            double deltaMax,
            Random random)
        {
            _modelDim = modelDim;
            _stateDim = stateDim;
            _innerDim = innerDim;

            var scale = NumOps.FromDouble(0.02);

            // SSM parameters
            _A = new Tensor<T>(new[] { innerDim, stateDim });
            _B = new Tensor<T>(new[] { innerDim, stateDim });
            _C = new Tensor<T>(new[] { innerDim, stateDim });
            _D = new Tensor<T>(new[] { innerDim });

            // Initialize A with negative values (for stability)
            for (int i = 0; i < _A.Length; i++)
            {
                _A[i] = NumOps.Negate(NumOps.FromDouble(1.0 + random.NextDouble() * 4.0));
            }

            InitializeWeights(_B, scale, random);
            InitializeWeights(_C, scale, random);

            AiDotNetEngine.Current.TensorFill(_D, NumOps.One);

            // Projections
            _inProj = new Tensor<T>(new[] { modelDim, innerDim * 2 });
            _outProj = new Tensor<T>(new[] { innerDim, modelDim });
            InitializeWeights(_inProj, scale, random);
            InitializeWeights(_outProj, scale, random);

            // Convolution
            _convWeight = new Tensor<T>(new[] { innerDim, convKernelSize });
            InitializeWeights(_convWeight, scale, random);

            // Delta projection -- Mamba's dt_init (Gu & Dao 2023, §3.6 and the reference
            // implementation): draw the discretization step log-uniformly from
            // [DeltaMin, DeltaMax], then store its INVERSE SOFTPLUS, because Forward applies
            // Softplus to this tensor. The constant 0.01 that stood here is the geometric mean
            // of the declared 0.001 and 0.1 -- the midpoint of the range both options describe,
            // hardcoded so that neither of them did anything.
            _deltaProj = new Tensor<T>(new[] { innerDim });
            double logMin = Math.Log(deltaMin);
            double logMax = Math.Log(deltaMax);
            for (int i = 0; i < _deltaProj.Length; i++)
            {
                double dt = Math.Exp(logMin + (random.NextDouble() * (logMax - logMin)));
                // inverse softplus, written as log(expm1(dt)) so small dt does not cancel.
                double inverseSoftplus = dt < 1e-3
                    ? Math.Log(dt) + (dt / 2.0)
                    : Math.Log(Math.Exp(dt) - 1.0);
                _deltaProj[i] = NumOps.FromDouble(inverseSoftplus);
            }
        }

        private static void InitializeWeights(Tensor<T> tensor, T scale, Random random)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = NumOps.Multiply(NumOps.FromDouble(random.NextGaussian()), scale);
            }
        }

        public Tensor<T> Forward(Tensor<T> input, int batchSize, int seqLen, int dim)
        {
            var engine = AiDotNetEngine.Current;
            int innerDim = _innerDim;
            int stateDim = _stateDim;
            int rows = batchSize * seqLen;

            // Input projection: x -> (z, x'), for every position at once
            var x = TabularMetrics.Flat(input, rows * dim).Reshape(rows, dim);
            var projected = engine.TensorMatMul(x, _inProj);
            var gate = engine.Sigmoid(engine.TensorNarrow(projected, 1, 0, innerDim)).Reshape(batchSize, seqLen, innerDim);
            var xPrime = engine.TensorNarrow(projected, 1, innerDim, innerDim).Reshape(batchSize, seqLen, innerDim);

            // Discretization: delta = softplus(deltaProj) = log(1 + exp(deltaProj)); a = exp(A * delta), b = B * delta
            var delta = engine.TensorLog(engine.TensorAddScalar(engine.TensorExp(_deltaProj), NumOps.One))
                .Reshape(innerDim, 1);
            var decay = engine.TensorExp(engine.TensorMultiply(_A.Reshape(innerDim, stateDim), delta))
                .Reshape(1, innerDim, stateDim);
            var inputGain = engine.TensorMultiply(_B.Reshape(innerDim, stateDim), delta).Reshape(1, innerDim, stateDim);
            var readout = _C.Reshape(1, innerDim, stateDim);
            var skip = _D.Reshape(1, innerDim);

            // Simplified SSM scan over the sequence: h = a * h + b * x', y = (C . h + D * x') * z
            var state = new Tensor<T>(new[] { batchSize, innerDim, stateDim });
            var steps = new Tensor<T>[seqLen];
            for (int s = 0; s < seqLen; s++)
            {
                var xs = engine.TensorNarrow(xPrime, 1, s, 1).Reshape(batchSize, innerDim);
                state = engine.TensorAdd(
                    engine.TensorMultiply(decay, state),
                    engine.TensorMultiply(inputGain, xs.Reshape(batchSize, innerDim, 1)));

                var hSum = engine.ReduceSum(engine.TensorMultiply(readout, state), new[] { 2 }, false);
                var y = engine.TensorAdd(hSum, engine.TensorMultiply(skip, xs));
                steps[s] = engine.TensorMultiply(y, engine.TensorNarrow(gate, 1, s, 1).Reshape(batchSize, innerDim))
                    .Reshape(batchSize, 1, innerDim);
            }

            // Output projection with residual connection
            var ySequence = engine.TensorConcatenate(steps, 1).Reshape(rows, innerDim);
            var output = engine.TensorAdd(x, engine.TensorMatMul(ySequence, _outProj));
            return output.Reshape(input._shape);
        }

        public void UpdateParameters(T learningRate)
        {
            // Simplified - in practice would use proper gradients
        }

        public void ResetState()
        {
            // Reset any cached state
        }
    }
}

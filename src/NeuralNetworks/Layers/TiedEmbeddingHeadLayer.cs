using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A language-model output head that shares its weights with the network's token embedding: the logits are
/// <c>h · Eᵀ</c>, where <c>E</c> is the embedding table <c>[vocabulary, hidden]</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is weight tying (Press &amp; Wolf, 2017), used by Llama, Gemma and most checkpoints that set
/// <c>tie_word_embeddings</c>. The head owns no weights: it reads the embedding's table on every forward, so a training
/// step moves one tensor, both uses contribute to its gradient, and the pair can never drift apart. Loading a tied
/// checkpoint into a separate dense head instead copies the table, which breaks the tie on the first update and holds
/// a second vocabulary × hidden matrix in memory.
/// </para>
/// <para>
/// The reference to the embedding layer is external state: it is not counted, optimized, cloned or saved here. The
/// network points it at the embedding's current instance whenever its layer list is built or replaced
/// (<see cref="BindToLayerGraph"/>), and the embedding's index in that list is what a saved model records.
/// </para>
/// <para><b>For Beginners:</b> The layer that turns words into vectors and the layer that turns vectors back into word
/// scores use the same table, as in the original model, so training the model trains both at once.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureFusion)]
// Like a dense projection: the last (hidden) axis becomes the vocabulary, every leading axis is carried through.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TiedEmbeddingHeadLayer<T> : LayerBase<T>, ILayerGraphBinding<T>, IShapeContract
{
    private int _sourceLayerIndex;
    private readonly int _vocabularySize;
    private readonly int _embeddingDimension;

    // A constant multiplier on the logits, such as Cohere's logit_scale. Applied at runtime, as Hugging Face does: baking it
    // into the weights would scale the shared embedding table too.
    private readonly double _logitScale;

    [Scratch]
    private readonly Tensor<T>? _logitScaleTensor;

    [ExternalState]
    private EmbeddingLayer<T>? _embedding;

    // Inference-only int8 copy of the table (GGUF Q8_0, laid out [vocabulary, hidden] = the block-Q8 kernel's [N, K]).
    // Valid only while the table it was made from is untouched; see IsQuantizedTableCurrent.
    private sbyte[]? _tableQ8;
    private float[]? _tableScalesQ8;
    [Scratch]
    private Tensor<T>? _tableAtQuantization;
    private int _tableVersionAtQuantization;

    /// <summary>Creates a head tied to <paramref name="embedding"/>.</summary>
    /// <param name="embedding">The token embedding whose table this head shares.</param>
    /// <param name="logitScale">A constant multiplier on the logits (1 for none).</param>
    public TiedEmbeddingHeadLayer(EmbeddingLayer<T> embedding, double logitScale = 1.0)
        : base(new[] { -1, embedding?.EmbeddingDimension ?? throw new ArgumentNullException(nameof(embedding)) },
               new[] { -1, embedding.VocabularySize })
    {
        _embedding = embedding;
        _vocabularySize = embedding.VocabularySize;
        _embeddingDimension = embedding.EmbeddingDimension;
        _sourceLayerIndex = -1;
        (_logitScale, _logitScaleTensor) = CreateLogitScale(logitScale);
    }

    /// <summary>
    /// Recreates a saved head, unbound. The owning network binds it to the embedding at
    /// <paramref name="sourceLayerIndex"/> once its layer list is complete.
    /// </summary>
    /// <param name="sourceLayerIndex">The embedding layer's index in the network's layer list.</param>
    /// <param name="vocabularySize">The embedding's vocabulary size.</param>
    /// <param name="embeddingDimension">The embedding's width.</param>
    /// <param name="logitScale">A constant multiplier on the logits (1 for none).</param>
    public TiedEmbeddingHeadLayer(
        [LayerState] int sourceLayerIndex,
        [LayerState] int vocabularySize,
        [LayerState] int embeddingDimension,
        [LayerState] double logitScale = 1.0)
        : base(new[] { -1, embeddingDimension }, new[] { -1, vocabularySize })
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        _sourceLayerIndex = sourceLayerIndex;
        _vocabularySize = vocabularySize;
        _embeddingDimension = embeddingDimension;
        (_logitScale, _logitScaleTensor) = CreateLogitScale(logitScale);
    }

    private static (double, Tensor<T>?) CreateLogitScale(double logitScale)
    {
        if (!(logitScale > 0.0) || double.IsInfinity(logitScale))
            throw new ArgumentOutOfRangeException(nameof(logitScale), logitScale, "The logit scale must be positive and finite.");
        if (logitScale == 1.0) return (logitScale, null);
        var tensor = new Tensor<T>(new[] { 1 });
        tensor[0] = MathHelper.GetNumericOperations<T>().FromDouble(logitScale);
        return (logitScale, tensor);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>The last axis becomes the vocabulary; every leading axis passes through, as for a dense projection.</summary>
    /// <param name="inputRank">The input's rank.</param>
    /// <returns>The output axes, or <c>null</c> for a rank this layer does not declare.</returns>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var vocabulary = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_vocabularySize));
        OutputAxisContract Pass(TensorAxis a) => new(a, AxisRelation.Same(a));
        return inputRank switch
        {
            1 => new[] { vocabulary },
            2 => new[] { Pass(TensorAxis.Batch), vocabulary },
            3 => new[] { Pass(TensorAxis.Batch), Pass(TensorAxis.Time), vocabulary },
            _ => null,
        };
    }

    /// <summary>The embedding layer this head reads, or <c>null</c> while unbound.</summary>
    internal EmbeddingLayer<T>? Embedding => _embedding;

    /// <inheritdoc/>
    void ILayerGraphBinding<T>.BindToLayerGraph(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));

        // The embedding this head already holds is authoritative when it is part of this graph (construction); the
        // index is then recorded for saving. Otherwise the graph was replaced (clone, deserialize) and the head still
        // points at the previous graph's embedding, or at nothing: resolve through the recorded index instead.
        if (_embedding is not null)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (ReferenceEquals(layers[i], _embedding))
                {
                    _sourceLayerIndex = i;
                    return;
                }
            }
        }

        if (_sourceLayerIndex < 0 || _sourceLayerIndex >= layers.Count)
        {
            throw new InvalidOperationException(
                $"TiedEmbeddingHeadLayer cannot find its embedding: recorded index {_sourceLayerIndex} is outside the " +
                $"network's {layers.Count} layers.");
        }

        if (layers[_sourceLayerIndex] is not EmbeddingLayer<T> embedding)
        {
            throw new InvalidOperationException(
                $"TiedEmbeddingHeadLayer expects an EmbeddingLayer at index {_sourceLayerIndex}, but found " +
                $"{layers[_sourceLayerIndex].GetType().Name}.");
        }

        if (embedding.VocabularySize != _vocabularySize || embedding.EmbeddingDimension != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"TiedEmbeddingHeadLayer was saved for a [{_vocabularySize}, {_embeddingDimension}] embedding, but the " +
                $"embedding at index {_sourceLayerIndex} is [{embedding.VocabularySize}, {embedding.EmbeddingDimension}].");
        }

        _embedding = embedding;
    }

    /// <summary>
    /// Installs an inference-only int8 copy of the shared table from GGUF Q8_0 blocks laid out
    /// <c>[vocabulary, hidden]</c>. It is dropped for good as soon as the table may change (see the forward pass).
    /// </summary>
    /// <param name="qs">The Q8_0 int8 payload, <c>vocabulary × hidden</c> values.</param>
    /// <param name="scales">One scale per 32-value block.</param>
    internal void SetQuantizedTableQ8_0(sbyte[] qs, float[] scales)
    {
        if (qs is null) throw new ArgumentNullException(nameof(qs));
        if (scales is null) throw new ArgumentNullException(nameof(scales));
        if (_embeddingDimension % 32 != 0)
            throw new ArgumentException($"embedding width ({_embeddingDimension}) must be a multiple of 32 for Q8_0.");
        if (qs.Length != (long)_vocabularySize * _embeddingDimension || scales.Length != (long)_vocabularySize * (_embeddingDimension / 32))
            throw new ArgumentException("Q8_0 payload does not match the embedding table's shape.");
        var table = RequireEmbedding().GetMaterializedEmbeddingTable();
        _tableQ8 = qs;
        _tableScalesQ8 = scales;
        _tableAtQuantization = table;
        _tableVersionAtQuantization = table.Version;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var table = RequireEmbedding().GetMaterializedEmbeddingTable();
        int[] shape = input.Shape.ToArray();
        int hidden = shape[shape.Length - 1];
        if (hidden != _embeddingDimension)
            throw new ArgumentException($"TiedEmbeddingHeadLayer expects a last dimension of {_embeddingDimension}, got {hidden}.", nameof(input));
        int rows = 1;
        for (int i = 0; i < shape.Length - 1; i++) rows *= shape[i];

        // Any sign that the weights are being trained or traced for training retires the int8 copy for good: after a
        // training step it would describe the pre-training table.
        bool training = IsTrainingMode || AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
            || AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive;
        if (training) DropQuantizedTable();

        var flat = Engine.Reshape(input, [rows, hidden]);
        Tensor<T> logits;
        if (!training && DenseLayer<T>.Q8SpeedFavorable && typeof(T) == typeof(float) && rows <= 16 && IsQuantizedTableCurrent(table)
            && _tableQ8 is { } q8 && _tableScalesQ8 is { } scales)
        {
            var inF = (float[])(object)flat.ToArray();
            var outArr = new float[rows * _vocabularySize];
            AiDotNet.Tensors.Engines.Simd.Q8BlockGemm.MatMul(inF, q8, scales, outArr, rows, hidden, _vocabularySize);
            logits = new Tensor<T>(new[] { rows, _vocabularySize }, new Vector<T>((T[])(object)outArr));
        }
        else
        {
            logits = Engine.TensorMatMulTransposed(flat, table);
        }

        // Tape-tracked element-wise multiply; TensorMultiplyScalar would not propagate gradient to the logits.
        if (_logitScaleTensor is not null) logits = Engine.TensorMultiply(logits, _logitScaleTensor);

        var outShape = shape.ToArray();
        outShape[outShape.Length - 1] = _vocabularySize;
        return Engine.Reshape(logits, outShape);
    }

    private EmbeddingLayer<T> RequireEmbedding() =>
        _embedding ?? throw new InvalidOperationException(
            "TiedEmbeddingHeadLayer is not bound to its embedding yet. It is bound when the owning network builds or " +
            "restores its layer list.");

    // The int8 copy is usable only while the table it came from is the same tensor, unmodified. A restore or clone can
    // install a different tensor; a write through the engine bumps the version.
    private bool IsQuantizedTableCurrent(Tensor<T> table)
    {
        if (_tableQ8 is null) return false;
        if (!ReferenceEquals(table, _tableAtQuantization) || table.Version != _tableVersionAtQuantization)
        {
            DropQuantizedTable();
            return false;
        }
        return true;
    }

    private void DropQuantizedTable()
    {
        _tableQ8 = null;
        _tableScalesQ8 = null;
        _tableAtQuantization = null;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => new Vector<T>(0);

    /// <inheritdoc/>
    public override void ClearGradients() { base.ClearGradients(); }

    /// <inheritdoc/>
    public override void ResetState() { /* no state of its own */ }
}
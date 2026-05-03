using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Base class for conditioning modules that compose other conditioners rather than
/// owning their own learnable weights. Examples: <see cref="DualTextConditioner{T}"/>
/// and <see cref="TripleTextConditioner{T}"/>, which delegate encoding work to one or
/// more inner CLIP / T5 / OpenCLIP encoders and then merge their outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This base intentionally does not allocate token / position / transformer weights
/// (unlike <see cref="TextConditioningBase{T}"/>, whose constructor sizes weight
/// vectors against a single transformer). Composite conditioners hold their inner
/// conditioners as fields and forward calls to them, so all that is shared between
/// composites is the boilerplate every conditioner needs:
/// </para>
/// <list type="bullet">
///   <item>The hardware-accelerated <see cref="IEngine"/> instance accessor used for
///   tensor merging operations like concat / add / matmul on the merged outputs.</item>
///   <item>The <see cref="INumericOperations{T}"/> dispatcher for type-generic
///   arithmetic.</item>
/// </list>
/// <para>
/// The <see cref="Engine"/> property is exposed as a protected instance member to
/// match the convention used by the rest of AiDotNet's base classes (see
/// <c>NeuralNetworkBase</c>, <c>AdversarialAttackBase</c>, <c>AugmentationBase</c>,
/// etc.). Subclasses MUST route all engine-dispatched tensor operations through this
/// property rather than calling <c>AiDotNetEngine.Current</c> directly, so that
/// future per-instance engine overrides remain a single-point change.
/// </para>
/// </remarks>
public abstract class CompositeConditioningBase<T> : IConditioningModule<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Hardware-accelerated engine for vector/tensor operations. Subclasses MUST
    /// route all engine dispatched operations through this property rather than
    /// calling <c>AiDotNetEngine.Current</c> directly.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <inheritdoc />
    public abstract int EmbeddingDimension { get; }

    /// <inheritdoc />
    public abstract ConditioningType ConditioningType { get; }

    /// <inheritdoc />
    public abstract bool ProducesPooledOutput { get; }

    /// <inheritdoc />
    public abstract int MaxSequenceLength { get; }

    /// <inheritdoc />
    public abstract Tensor<T> Encode(Tensor<T> input);

    /// <inheritdoc />
    public abstract Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null);

    /// <inheritdoc />
    public abstract Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings);

    /// <inheritdoc />
    public abstract Tensor<T> GetUnconditionalEmbedding(int batchSize);

    /// <inheritdoc />
    public abstract Tensor<T> Tokenize(string text);

    /// <inheritdoc />
    public abstract Tensor<T> TokenizeBatch(string[] texts);
}

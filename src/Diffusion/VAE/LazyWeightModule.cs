using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// A weight whose shape is known at construction but whose storage is allocated on first use, so a paper-scale model
/// can be built, counted and described without paying for its weights.
/// </summary>
/// <typeparam name="T">The numeric type of the weight.</typeparam>
/// <remarks>
/// The initial values come from a seed drawn when the weight is declared, inside the model's seed scope. Allocating
/// later, in any order, therefore gives the same values a model built eagerly would have.
/// </remarks>
internal sealed class LazyWeight<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Action<Tensor<T>, Random> _initialize;
    private readonly int _seed;
    private Tensor<T>? _value;

    public LazyWeight(int[] shape, Action<Tensor<T>, Random> initialize)
    {
        Shape = shape;
        long count = 1;
        foreach (int dimension in shape)
        {
            if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(shape), "Every dimension must be positive.");
            count *= dimension;
        }
        Count = count;
        _initialize = initialize;
        _seed = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom().Next();
    }

    /// <summary>The weight's shape.</summary>
    public int[] Shape { get; }

    /// <summary>The number of values, known without allocating.</summary>
    public long Count { get; }

    /// <summary>Whether storage has been allocated.</summary>
    public bool IsMaterialized => _value is not null;

    /// <summary>The live tensor; allocated and initialized on first access.</summary>
    public Tensor<T> Value
    {
        get
        {
            if (_value is null)
            {
                var tensor = new Tensor<T>(Shape);
                _initialize(tensor, RandomHelper.CreateSeededRandom(_seed));
                _value = tensor;
            }
            return _value;
        }
    }

    /// <summary>Uniform in +-1/sqrt(fan_in): torch.nn.Conv*d and Linear's default initialization.</summary>
    public static LazyWeight<T> Uniform(int[] shape, int fanIn) => new(shape, (tensor, random) =>
    {
        double bound = 1.0 / Math.Sqrt(fanIn);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * bound);
    });

    /// <summary>A constant fill (a norm's scale of one or shift of zero).</summary>
    public static LazyWeight<T> Constant(int[] shape, double value) => new(shape, (tensor, _) =>
    {
        if (value != 0) tensor.Fill(NumOps.FromDouble(value));
    });
}

/// <summary>
/// A building block whose own weights are <see cref="LazyWeight{T}"/>s: its parameter count and layout are known at
/// construction, and its storage is allocated on first forward or when the model's parameter registry asks for values.
/// </summary>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal abstract class LazyWeightModule<T> : IParameterSource<T>, IParameterChunkSource<T>, IParameterLayoutSource,
    IParameterMaterializationSource
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>This block's own weights, in a fixed order.</summary>
    protected virtual IEnumerable<LazyWeight<T>> OwnWeights() => Array.Empty<LazyWeight<T>>();

    /// <summary>The child blocks, in a fixed order; null entries (an optional block not built) are skipped.</summary>
    protected virtual IEnumerable<LazyWeightModule<T>?> Children() => Array.Empty<LazyWeightModule<T>?>();

    private IEnumerable<LazyWeightModule<T>> PresentChildren()
    {
        foreach (var child in Children())
            if (child is not null) yield return child;
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var weight in OwnWeights()) total += weight.Count;
            foreach (var child in PresentChildren()) total += child.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    public void MaterializeParameters()
    {
        foreach (var weight in OwnWeights()) _ = weight.Value;
        foreach (var child in PresentChildren()) child.MaterializeParameters();
    }

    /// <inheritdoc />
    /// <remarks>Describes every slot without allocating: an unallocated weight is shape-resolved but unmaterialized.</remarks>
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var slots = new List<ParameterSlotDescriptor>();
        int own = 0;
        foreach (var weight in OwnWeights())
        {
            slots.Add(new ParameterSlotDescriptor(
                $"w{own}", ParameterSlotRole.Trainable,
                weight.IsMaterialized ? ParameterReadiness.Materialized : ParameterReadiness.ShapeResolvedUnmaterialized,
                weight.Count, shape: weight.Shape, elementType: typeof(T).FullName,
                materializedParameterCount: weight.IsMaterialized ? weight.Count : 0));
            own++;
        }
        int index = 0;
        foreach (var child in PresentChildren())
        {
            string prefix = ParameterStableId.IndexSegment(index);
            foreach (var slot in child.GetParameterLayout())
            {
                slots.Add(new ParameterSlotDescriptor(
                    prefix + "/" + slot.StableId, slot.Role, slot.Readiness, slot.ParameterCount,
                    shape: slot.Shape, elementType: slot.ElementType, updatePolicy: slot.UpdatePolicy,
                    persistence: slot.Persistence, ownership: slot.Ownership, availability: slot.Availability,
                    materializedParameterCount: slot.MaterializedParameterCount));
            }
            index++;
        }
        return slots;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>(checked((int)ParameterCount));
        int offset = 0;
        foreach (var tensor in LiveTensors())
            for (int i = 0; i < tensor.Length; i++) result[offset++] = tensor[i];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"{GetType().Name} expects {ParameterCount} parameter values but received {parameters.Length}.", nameof(parameters));
        int offset = 0;
        // Written through, never replaced: the forward pass keeps reading these instances.
        foreach (var tensor in LiveTensors())
            for (int i = 0; i < tensor.Length; i++) tensor[i] = parameters[offset++];
    }

    /// <inheritdoc />
    public IEnumerable<ParameterChunk<T>> GetParameterStateChunks()
    {
        int own = 0;
        foreach (var weight in OwnWeights())
        {
            yield return new ParameterChunk<T>($"w{own}", ParameterSlotRole.Trainable, weight.Value);
            own++;
        }
        int index = 0;
        foreach (var child in PresentChildren())
        {
            string prefix = ParameterStableId.IndexSegment(index);
            foreach (var chunk in child.GetParameterStateChunks())
                yield return new ParameterChunk<T>(prefix + "/" + chunk.StableId, chunk.Role, chunk.Tensor, chunk.SourceTensor, chunk.IsWritableInPlace);
            index++;
        }
    }

    // Every weight's live tensor, own then children, allocating any that are not yet.
    private IEnumerable<Tensor<T>> LiveTensors()
    {
        foreach (var weight in OwnWeights()) yield return weight.Value;
        foreach (var child in PresentChildren())
            foreach (var tensor in child.LiveTensors()) yield return tensor;
    }
}

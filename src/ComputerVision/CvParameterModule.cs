using AiDotNet.Models.Parameters;

namespace AiDotNet.ComputerVision;

/// <summary>
/// Base for the hand-rolled building blocks of the computer-vision detection and OCR models - an
/// encoder layer, a cross-attention block, a detection head - that own weights but are neither a
/// <c>LayerBase</c> nor a <c>ModelBase</c>.
/// </summary>
/// <remarks>
/// <para>
/// The parameter generator registers a model field only when its type is a parameter source, a layer
/// or layer collection, or follows the <c>EnumerateLayers()</c> convention. A field typed as a plain
/// helper class matched none of those, so everything behind it was invisible: missing from
/// <c>GetParameters()</c>, from <c>Serialize</c>, from the rebuild-and-reload <c>DeepCopy</c>, and from
/// training. For DETR that was the entire encoder and decoder.
/// </para>
/// <para>
/// Deriving from this class makes a block a parameter source in its own right, so the generator picks
/// it up, and the block exposes its weights as LIVE chunks - the very tensor instances its forward pass
/// reads. That matters for training: the autodiff tape keys gradients by reference, so an optimizer
/// can only update a weight if it is handed that exact instance.
/// </para>
/// <para>
/// A derived block declares, in a fixed order, the child components it owns
/// (<see cref="ParameterChildren"/>) and any raw weight tensors it holds directly
/// (<see cref="OwnParameterTensors"/>). Every child must itself expose live chunks; a child that can
/// only produce a copy is rejected at first use, because silently training a copy would leave the real
/// weight untouched.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal abstract class CvParameterModule<T> : IParameterSource<T>, IParameterChunkSource<T>
{
    /// <summary>
    /// The child components this block owns, in a fixed order. Null entries (an optional component
    /// the configuration did not build) are skipped.
    /// </summary>
    protected abstract IEnumerable<IParameterSource<T>?> ParameterChildren();

    /// <summary>
    /// Raw weight tensors this block holds directly (a learnable query embedding, a norm's scale and
    /// shift), in a fixed order. They are exposed and restored in place.
    /// </summary>
    protected virtual IEnumerable<Tensor<T>> OwnParameterTensors() => Array.Empty<Tensor<T>>();

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var tensor in OwnParameterTensors())
            {
                total += tensor.Length;
            }

            foreach (var child in Children())
            {
                total += child.ParameterCount;
            }

            return total;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>(checked((int)ParameterCount));
        int offset = 0;
        foreach (var tensor in OwnParameterTensors())
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result[offset++] = tensor[i];
            }
        }

        foreach (var child in Children())
        {
            var values = child.GetParameters();
            for (int i = 0; i < values.Length; i++)
            {
                result[offset++] = values[i];
            }
        }

        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        long expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"{GetType().Name} expects {expected} parameter values but received {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;
        foreach (var tensor in OwnParameterTensors())
        {
            // Written through, never replaced: the forward pass keeps reading this instance.
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = parameters[offset++];
            }
        }

        foreach (var child in Children())
        {
            int count = checked((int)child.ParameterCount);
            var slice = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                slice[i] = parameters[offset++];
            }

            child.SetParameters(slice);
        }
    }

    /// <inheritdoc />
    public IEnumerable<ParameterChunk<T>> GetParameterStateChunks()
    {
        int own = 0;
        foreach (var tensor in OwnParameterTensors())
        {
            if (tensor.Length > 0)
            {
                yield return new ParameterChunk<T>($"w{own}", ParameterSlotRole.Trainable, tensor);
            }

            own++;
        }

        int index = 0;
        foreach (var child in Children())
        {
            if (child is not IParameterChunkSource<T> chunked)
            {
                throw new InvalidOperationException(
                    $"{GetType().Name} child #{index} ({child.GetType().Name}) exposes parameters only as a "
                    + "copy. It must implement IParameterChunkSource<T> so training updates the live weight.");
            }

            foreach (var chunk in chunked.GetParameterStateChunks())
            {
                string id = chunk.StableId == "$" ? $"{index}" : $"{index}/{chunk.StableId}";
                yield return new ParameterChunk<T>(id, chunk.Role, chunk.Tensor, chunk.SourceTensor, chunk.IsWritableInPlace);
            }

            index++;
        }
    }

    private IEnumerable<IParameterSource<T>> Children()
    {
        foreach (var child in ParameterChildren())
        {
            if (child is not null)
            {
                yield return child;
            }
        }
    }
}

/// <summary>
/// A <see cref="CvParameterModule{T}"/> whose children and own tensors are supplied by delegates.
/// </summary>
/// <remarks>
/// For public building blocks (the region proposal network, for one) that cannot derive from the
/// internal <see cref="CvParameterModule{T}"/>: they hold one of these and forward
/// <see cref="IParameterSource{T}"/> and <see cref="IParameterChunkSource{T}"/> to it.
/// </remarks>
/// <typeparam name="T">The numeric type of the weights.</typeparam>
internal sealed class DelegatingCvParameterModule<T> : CvParameterModule<T>
{
    private readonly Func<IEnumerable<IParameterSource<T>?>> _children;
    private readonly Func<IEnumerable<Tensor<T>>>? _own;

    /// <summary>Creates a module over the given children and, optionally, raw tensors.</summary>
    public DelegatingCvParameterModule(
        Func<IEnumerable<IParameterSource<T>?>> children,
        Func<IEnumerable<Tensor<T>>>? own = null)
    {
        _children = children ?? throw new ArgumentNullException(nameof(children));
        _own = own;
    }

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => _children();

    /// <inheritdoc />
    protected override IEnumerable<Tensor<T>> OwnParameterTensors() => _own?.Invoke() ?? Array.Empty<Tensor<T>>();
}

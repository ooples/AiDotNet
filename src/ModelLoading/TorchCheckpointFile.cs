using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.ModelLoading;

/// <summary>
/// One released checkpoint opened as a <see cref="TorchTensorSource"/>: a <c>.safetensors</c> file is read through its
/// stream and decoded chunk by chunk into the loader's storage; a PyTorch <c>.pt</c>/<c>.pth</c>/<c>.tar</c>/<c>.bin</c> state dict is
/// unpickled by <see cref="ComputerVision.Weights.WeightLoader"/>. Dispose it when loading is done: the file stays open
/// until then.
/// </summary>
/// <remarks>A weight-norm parametrization's names (<c>parametrizations.weight.original0/1</c>) also find a checkpoint saved
/// with the older <c>weight_g</c>/<c>weight_v</c> spelling.</remarks>
internal sealed class TorchCheckpointFile : TorchTensorSource, IDisposable
{
    private readonly string _path;
    private readonly FileStream? _stream;
    private readonly SafetensorsFile? _safetensors;
    private readonly Dictionary<string, Tensor<float>>? _state;
    private readonly Pretrained.ShardedSafetensorsSource? _shards;
    private bool _strictAxes;

    private TorchCheckpointFile(string path, FileStream? stream, SafetensorsFile? safetensors, Dictionary<string, Tensor<float>>? state,
        Pretrained.ShardedSafetensorsSource? shards = null)
    {
        _shards = shards;
        _path = path;
        _stream = stream;
        _safetensors = safetensors;
        _state = state;
    }

    /// <summary>Opens a checkpoint by its extension (a torch.save archive under another name is recognized).</summary>
    /// <param name="path">The checkpoint.</param>
    /// <param name="strictAxes">Whether a shape the loader gives must equal the stored axes (otherwise only the counts
    /// must agree).</param>
    public static TorchCheckpointFile Open(string path, bool strictAxes = false)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint {path} is missing.", path);
        if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".ckpt", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            || (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) && IsZipArchive(path)))
            return new TorchCheckpointFile(path, null, null, new ComputerVision.Weights.WeightLoader().LoadWeights(path)) { _strictAxes = strictAxes };
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess);
        try
        {
            return new TorchCheckpointFile(path, stream, SafetensorsReader.Read(stream), null) { _strictAxes = strictAxes };
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    // torch.save writes a zip archive under any name (fairseq's vocoders as g_00500000).
    private static bool IsZipArchive(string path)
    {
        using var stream = File.OpenRead(path);
        return stream.ReadByte() == 'P' && stream.ReadByte() == 'K';
    }

    /// <summary>Opens a Hugging Face model directory's safetensors weights (its <c>model.safetensors.index.json</c> shards,
    /// else <c>model.safetensors</c>, else every <c>*.safetensors</c> file) as one source.</summary>
    public static TorchCheckpointFile OpenDirectory(string directory, bool strictAxes = false) =>
        new(directory, null, null, null, Pretrained.ShardedSafetensorsSource.Open(directory)) { _strictAxes = strictAxes };

    /// <inheritdoc />
    protected override string Description => $"the checkpoint {_path}";

    /// <summary>Every stored tensor's name.</summary>
    public IReadOnlyCollection<string> Names =>
        _state is not null ? _state.Keys : _safetensors is not null ? _safetensors.Names : _shards!.TensorNames;

    private readonly HashSet<string> _used = new(StringComparer.Ordinal);

    /// <summary>Throws when a stored tensor was never read and <paramref name="unused"/> does not expect that.</summary>
    /// <param name="what">The checkpoint's name in the message.</param>
    /// <param name="unused">Whether a stored name is expected to go unread (buffers, modules the model does not build).</param>
    public void RequireAllUsed(string what, Func<string, bool>? unused = null)
    {
        var left = Names.Where(k => !_used.Contains(k) && (unused is null || !unused(k))).ToList();
        if (left.Count > 0)
            throw new InvalidDataException($"The {what} checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
    }

    /// <inheritdoc />
    public override bool Contains(string name) => Stored(name) || Stored(LegacyWeightNormName(name));

    private bool Stored(string name) => _state is not null ? _state.ContainsKey(name) : FileOf(name)?.Get(name) is not null;

    // The safetensors file holding the tensor: the one file, or its shard.
    private SafetensorsFile? FileOf(string name) => _safetensors ?? _shards?.FileOf(name);

    private string Resolve(string name) => Stored(name) ? name : LegacyWeightNormName(name);

    private static string LegacyWeightNormName(string name) => name
        .Replace(".parametrizations.weight.original0", ".weight_g")
        .Replace(".parametrizations.weight.original1", ".weight_v");

    /// <inheritdoc />
    public override long[]? ShapeOf(string name)
    {
        name = Resolve(name);
        return _state is not null
            ? _state.TryGetValue(name, out var t) ? t.Shape.ToArray().Select(d => (long)d).ToArray() : null
            : FileOf(name)?.Get(name)?.Shape.ToArray();
    }

    /// <inheritdoc />
    protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink)
    {
        if (_strictAxes && shape.Length > 0 && ShapeOf(name) is { Length: > 0 } axes && !axes.SequenceEqual(shape.Select(d => (long)d)))
            throw new InvalidDataException($"'{name}' is [{string.Join(", ", axes)}] in {Description} but [{string.Join(", ", shape)}] in this model; "
                + "build the model with the checkpoint's configuration.");
        name = Resolve(name);
        _used.Add(name);
        if (_state is null)
        {
            var file = FileOf(name);
            var entry = file?.Get(name) ?? throw new InvalidDataException($"{Description} has no tensor '{name}'.");
            if (entry.DataType == "I64")
            {
                // An int64 scalar such as num_batches_tracked.
                sink(0, new[] { (double)BitConverter.ToInt64(file!.GetRawBytes(name), 0) });
                return 1;
            }
            long count = 0;
            file!.Decode(name, (start, values) =>
            {
                sink(start, values);
                count += values.Length;
            });
            return count;
        }
        var tensor = _state.TryGetValue(name, out var t) ? t : throw new InvalidDataException($"{Description} has no tensor '{name}'.");
        return DecodeFloats(tensor.AsSpan(), sink);
    }

    public void Dispose()
    {
        _stream?.Dispose();
        _shards?.Dispose();
    }
}

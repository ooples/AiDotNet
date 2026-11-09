using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// One released checkpoint opened for loading: a <c>.safetensors</c> file is read through its stream, one tensor at a
/// time, so a multi-gigabyte file is never held in memory whole; a PyTorch <c>.pt</c>/<c>.pth</c> state dict is
/// materialized by <see cref="ComputerVision.Weights.WeightLoader"/>, which pickled files need. Dispose it when loading is
/// done: the file stays open until then.
/// </summary>
internal sealed class TorchCheckpointFile : IDisposable
{
    private readonly string _path;
    private readonly FileStream? _stream;
    private readonly SafetensorsFile? _safetensors;
    private readonly Dictionary<string, Tensor<float>>? _state;

    private TorchCheckpointFile(string path, FileStream? stream, SafetensorsFile? safetensors, Dictionary<string, Tensor<float>>? state)
    {
        _path = path;
        _stream = stream;
        _safetensors = safetensors;
        _state = state;
    }

    /// <summary>Opens a checkpoint by its extension.</summary>
    public static TorchCheckpointFile Open(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint {path} is missing.", path);
        if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase)
            || path.EndsWith(".tar", StringComparison.OrdinalIgnoreCase))
            return new TorchCheckpointFile(path, null, null, new ComputerVision.Weights.WeightLoader().LoadWeights(path));
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess);
        try
        {
            return new TorchCheckpointFile(path, stream, SafetensorsReader.Read(stream), null);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>Whether the checkpoint has a tensor of this name.</summary>
    public bool Contains(string name) => _safetensors is not null ? _safetensors.Get(name) is not null : _state!.ContainsKey(name);

    /// <summary>The tensor's shape, or null when absent.</summary>
    public long[]? Shape(string name) => _safetensors is not null
        ? _safetensors.Get(name)?.Shape.ToArray()
        : _state!.TryGetValue(name, out var t) ? t.Shape.ToArray().Select(d => (long)d).ToArray() : null;

    /// <summary>The tensor's values (an int64 scalar such as <c>num_batches_tracked</c> as its value), or null when absent.</summary>
    public double[]? TryRead(string name)
    {
        if (_safetensors is not null)
        {
            var entry = _safetensors.Get(name);
            if (entry is null) return null;
            if (entry.DataType == "I64") return new[] { (double)BitConverter.ToInt64(_safetensors.GetRawBytes(name), 0) };
            return _safetensors.ReadAsDouble(name);
        }
        return _state!.TryGetValue(name, out var tensor) ? tensor.ToArray().Select(v => (double)v).ToArray() : null;
    }

    /// <summary>The tensor's values, its element count checked against <paramref name="shape"/> when one is given.</summary>
    public double[] Read(string name, int[] shape)
    {
        var values = TryRead(name) ?? throw new InvalidDataException($"The checkpoint {_path} has no tensor '{name}'.");
        if (shape.Length > 0 && values.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new InvalidDataException($"'{name}' has {values.Length} values in the checkpoint but [{string.Join(", ", shape)}] in this model; "
                + "build the model with the checkpoint's configuration.");
        return values;
    }

    public void Dispose() => _stream?.Dispose();
}

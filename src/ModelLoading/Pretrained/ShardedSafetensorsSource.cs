using AiDotNet.Agentic.Models.Local;
using Newtonsoft.Json.Linq;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// An <see cref="INamedTensorSource"/> over a Hugging Face model directory whose weights may be split
/// across several <c>*.safetensors</c> shards. Resolves the layout automatically:
/// a <c>model.safetensors.index.json</c> weight-map when present, otherwise a single
/// <c>model.safetensors</c>, otherwise every <c>*.safetensors</c> file in the directory.
/// </summary>
/// <remarks>
/// <para>
/// Each shard is opened stream-backed (tensor bytes are read on demand, never copied wholesale), so an
/// 8B-parameter multi-shard checkpoint is not materialized in memory. Dispose the source to close the
/// underlying file handles.
/// </para>
/// <para><b>For Beginners:</b> Big models are saved in several files. This class stitches those files back
/// together and lets you ask for any weight by name without caring which file it lives in.
/// </para>
/// </remarks>
public sealed class ShardedSafetensorsSource : INamedTensorSource, IDisposable
{
    private readonly List<FileStream> _streams = new();
    private readonly Dictionary<string, SafetensorsFile> _tensorToFile = new(StringComparer.Ordinal);

    private ShardedSafetensorsSource() { }

    /// <summary>Opens the safetensors weights in <paramref name="directory"/> as a single tensor source.</summary>
    /// <param name="directory">A Hugging Face model directory.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="directory"/> is null/empty.</exception>
    /// <exception cref="DirectoryNotFoundException">Thrown when the directory does not exist.</exception>
    /// <exception cref="FileNotFoundException">Thrown when no safetensors files are found.</exception>
    /// <exception cref="InvalidDataException">Thrown when the index weight-map references a missing shard.</exception>
    public static ShardedSafetensorsSource Open(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("Directory must be non-empty.", nameof(directory));
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Model directory not found: {directory}");

        var source = new ShardedSafetensorsSource();
        try
        {
            string indexPath = Path.Combine(directory, "model.safetensors.index.json");
            if (File.Exists(indexPath))
                source.LoadFromIndex(directory, indexPath);
            else
                source.LoadFromLooseShards(directory);
            return source;
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> TensorNames => _tensorToFile.Keys;

    /// <inheritdoc/>
    public double[] ReadAsDouble(string name)
    {
        if (!_tensorToFile.TryGetValue(name, out var file))
            throw new ArgumentException($"Tensor '{name}' is not present in this model directory.", nameof(name));
        return file.ReadAsDouble(name);
    }

    private void LoadFromIndex(string directory, string indexPath)
    {
        JObject index;
        try
        {
            index = JObject.Parse(File.ReadAllText(indexPath));
        }
        catch (Newtonsoft.Json.JsonException ex)
        {
            throw new InvalidDataException("model.safetensors.index.json is not valid JSON: " + ex.Message);
        }

        if (index["weight_map"] is not JObject weightMap)
            throw new InvalidDataException("model.safetensors.index.json is missing a 'weight_map' object.");

        // Open each referenced shard once, then map every tensor name to its shard.
        var openedShards = new Dictionary<string, SafetensorsFile>(StringComparer.Ordinal);
        foreach (var entry in weightMap)
        {
            var shardValue = (string?)entry.Value;
            if (shardValue is null || shardValue.Trim().Length == 0)
                throw new InvalidDataException($"weight_map entry '{entry.Key}' has no shard file.");
            string shardFile = shardValue;

            if (!openedShards.TryGetValue(shardFile, out var file))
            {
                string shardPath = Path.Combine(directory, shardFile);
                if (!File.Exists(shardPath))
                    throw new InvalidDataException($"index references missing shard '{shardFile}'.");
                file = OpenShard(shardPath);
                openedShards[shardFile] = file;
            }

            _tensorToFile[entry.Key] = file;
        }
    }

    private void LoadFromLooseShards(string directory)
    {
        string single = Path.Combine(directory, "model.safetensors");
        string[] shardPaths = File.Exists(single)
            ? new[] { single }
            : Directory.GetFiles(directory, "*.safetensors");

        if (shardPaths.Length == 0)
            throw new FileNotFoundException($"No .safetensors files found in {directory}.");

        foreach (var shardPath in shardPaths)
        {
            var file = OpenShard(shardPath);
            foreach (var name in file.Names)
                _tensorToFile[name] = file; // last writer wins on the rare duplicate; shards are disjoint in practice
        }
    }

    private SafetensorsFile OpenShard(string path)
    {
        var stream = File.OpenRead(path);
        _streams.Add(stream);
        return SafetensorsReader.Read(stream);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var stream in _streams)
            stream.Dispose();
        _streams.Clear();
        _tensorToFile.Clear();
    }
}

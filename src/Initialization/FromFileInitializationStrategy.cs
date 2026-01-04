using Newtonsoft.Json;

namespace AiDotNet.Initialization;

/// <summary>
/// Initialization strategy that loads weights from an external file.
/// </summary>
/// <remarks>
/// <para>
/// This strategy loads pre-trained weights from a file, enabling transfer learning
/// and model checkpointing. Weights are loaded during the first initialization call
/// and cached for subsequent layers.
/// </para>
/// <para><b>For Beginners:</b> Transfer learning is like giving your network a head start
/// by using weights that were already trained on a similar task. Instead of starting
/// from random values, you start with values that already know useful patterns.
/// </para>
/// <para>
/// Supported formats:
/// - JSON: Human-readable format with weight arrays
/// - Binary: Compact binary format for faster loading
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FromFileInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly string _filePath;
    private readonly WeightFileFormat _format;
    private Dictionary<string, T[]>? _loadedWeights;
    private Dictionary<string, T[]>? _loadedBiases;
    private int _weightLayerIndex;
    private int _biasLayerIndex;
    private readonly object _loadLock = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="FromFileInitializationStrategy{T}"/> class.
    /// </summary>
    /// <param name="filePath">The path to the weights file.</param>
    /// <param name="format">The format of the weights file. Default is Auto-detect.</param>
    public FromFileInitializationStrategy(string filePath, WeightFileFormat format = WeightFileFormat.Auto)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        _filePath = filePath;
        _format = format == WeightFileFormat.Auto ? DetectFormat(filePath) : format;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => true;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        EnsureWeightsLoaded();

        var key = $"weights_{_weightLayerIndex}";
        if (_loadedWeights is not null && _loadedWeights.TryGetValue(key, out var weightData))
        {
            if (weightData.Length != weights.Length)
            {
                throw new InvalidOperationException(
                    $"Weight size mismatch for layer {_weightLayerIndex}. Expected {weights.Length}, got {weightData.Length}.");
            }

            Array.Copy(weightData, weights.Data, weightData.Length);
            _weightLayerIndex++;
        }
        else
        {
            // Fall back to Xavier initialization if weights not found
            XavierNormalInitialize(weights, inputSize, outputSize);
            _weightLayerIndex++;
        }
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        EnsureWeightsLoaded();

        var key = $"biases_{_biasLayerIndex}";
        if (_loadedBiases is not null && _loadedBiases.TryGetValue(key, out var biasData))
        {
            if (biasData.Length != biases.Length)
            {
                throw new InvalidOperationException(
                    $"Bias size mismatch for layer {_biasLayerIndex}. Expected {biases.Length}, got {biasData.Length}.");
            }

            Array.Copy(biasData, biases.Data, biasData.Length);
            _biasLayerIndex++;
        }
        else
        {
            // Fall back to zero initialization if biases not found
            ZeroInitializeBiases(biases);
            _biasLayerIndex++;
        }
    }

    /// <summary>
    /// Resets the layer indices for a fresh initialization pass.
    /// </summary>
    /// <remarks>
    /// Call this method if you need to re-initialize a network with the same weights.
    /// </remarks>
    public void Reset()
    {
        _weightLayerIndex = 0;
        _biasLayerIndex = 0;
    }

    /// <summary>
    /// Clears the cached weights, forcing a reload on next initialization.
    /// </summary>
    public void ClearCache()
    {
        lock (_loadLock)
        {
            _loadedWeights = null;
            _loadedBiases = null;
            _weightLayerIndex = 0;
            _biasLayerIndex = 0;
        }
    }

    private void EnsureWeightsLoaded()
    {
        if (_loadedWeights is not null)
        {
            return;
        }

        lock (_loadLock)
        {
            if (_loadedWeights is not null)
            {
                return;
            }

            if (!File.Exists(_filePath))
            {
                throw new FileNotFoundException($"Weights file not found: {_filePath}");
            }

            switch (_format)
            {
                case WeightFileFormat.Json:
                    LoadFromJson();
                    break;
                case WeightFileFormat.Binary:
                    LoadFromBinary();
                    break;
                default:
                    throw new NotSupportedException($"Unsupported weight file format: {_format}");
            }
        }
    }

    private void LoadFromJson()
    {
        var json = File.ReadAllText(_filePath);
        var data = JsonConvert.DeserializeObject<WeightFileData>(json);

        if (data is null)
        {
            throw new InvalidOperationException("Failed to deserialize weights file.");
        }

        _loadedWeights = new Dictionary<string, T[]>();
        _loadedBiases = new Dictionary<string, T[]>();

        if (data.Weights is not null)
        {
            foreach (var kvp in data.Weights)
            {
                _loadedWeights[kvp.Key] = ConvertToT(kvp.Value);
            }
        }

        if (data.Biases is not null)
        {
            foreach (var kvp in data.Biases)
            {
                _loadedBiases[kvp.Key] = ConvertToT(kvp.Value);
            }
        }
    }

    private void LoadFromBinary()
    {
        using var stream = File.OpenRead(_filePath);
        using var reader = new BinaryReader(stream);

        // Read header
        var magic = reader.ReadInt32();
        if (magic != 0x574E4E41) // "ANNW" in little-endian
        {
            throw new InvalidOperationException("Invalid binary weights file format.");
        }

        var version = reader.ReadInt32();
        if (version != 1)
        {
            throw new NotSupportedException($"Unsupported binary weights file version: {version}");
        }

        _loadedWeights = new Dictionary<string, T[]>();
        _loadedBiases = new Dictionary<string, T[]>();

        // Read weights count
        var weightsCount = reader.ReadInt32();
        for (int i = 0; i < weightsCount; i++)
        {
            var key = reader.ReadString();
            var length = reader.ReadInt32();
            var values = new double[length];
            for (int j = 0; j < length; j++)
            {
                values[j] = reader.ReadDouble();
            }
            _loadedWeights[key] = ConvertToT(values);
        }

        // Read biases count
        var biasesCount = reader.ReadInt32();
        for (int i = 0; i < biasesCount; i++)
        {
            var key = reader.ReadString();
            var length = reader.ReadInt32();
            var values = new double[length];
            for (int j = 0; j < length; j++)
            {
                values[j] = reader.ReadDouble();
            }
            _loadedBiases[key] = ConvertToT(values);
        }
    }

    private T[] ConvertToT(double[] values)
    {
        var result = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = NumOps.FromDouble(values[i]);
        }
        return result;
    }

    private static WeightFileFormat DetectFormat(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".json" => WeightFileFormat.Json,
            ".bin" or ".weights" => WeightFileFormat.Binary,
            _ => WeightFileFormat.Json // Default to JSON
        };
    }

    /// <summary>
    /// Internal class for JSON serialization of weight data.
    /// </summary>
    private class WeightFileData
    {
        [JsonProperty("weights")]
        public Dictionary<string, double[]>? Weights { get; set; }

        [JsonProperty("biases")]
        public Dictionary<string, double[]>? Biases { get; set; }
    }
}

/// <summary>
/// Specifies the format of a weight file.
/// </summary>
public enum WeightFileFormat
{
    /// <summary>
    /// Auto-detect format based on file extension.
    /// </summary>
    Auto,

    /// <summary>
    /// JSON format with human-readable weight arrays.
    /// </summary>
    Json,

    /// <summary>
    /// Compact binary format for faster loading.
    /// </summary>
    Binary
}

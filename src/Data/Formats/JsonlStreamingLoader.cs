using Newtonsoft.Json.Linq;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Streams JSONL (JSON Lines) files line by line for efficient large-scale text data loading.
/// </summary>
/// <remarks>
/// <para>
/// JSONL is the standard format for LLM training data, where each line is a valid JSON object.
/// This loader reads files lazily without loading the entire file into memory, making it suitable
/// for datasets that are too large to fit in RAM.
/// </para>
/// <para>
/// Supports reading from multiple files, optional shuffling with a shuffle buffer,
/// and configurable field extraction.
/// </para>
/// </remarks>
public class JsonlStreamingLoader : IDisposable
{
    private readonly string[] _filePaths;
    private readonly string? _textField;
    private readonly string? _labelField;
    private readonly int _shuffleBufferSize;
    private readonly int? _maxSamples;
    private readonly int? _seed;
    private bool _disposed;

    /// <summary>
    /// Creates a new JSONL streaming loader.
    /// </summary>
    /// <param name="filePaths">Paths to the JSONL files.</param>
    /// <param name="textField">JSON field name containing the text data. If null, the entire line is returned.</param>
    /// <param name="labelField">JSON field name containing the label. If null, no labels are extracted.</param>
    /// <param name="shuffleBufferSize">Size of the shuffle buffer. 0 or 1 disables shuffling.</param>
    /// <param name="maxSamples">Optional maximum number of samples.</param>
    /// <param name="seed">Optional random seed for shuffling.</param>
    public JsonlStreamingLoader(
        string[] filePaths,
        string? textField = null,
        string? labelField = null,
        int shuffleBufferSize = 0,
        int? maxSamples = null,
        int? seed = null)
    {
        if (filePaths is null || filePaths.Length == 0)
            throw new ArgumentException("At least one file path is required.", nameof(filePaths));

        _filePaths = filePaths;
        _textField = textField;
        _labelField = labelField;
        _shuffleBufferSize = shuffleBufferSize;
        _maxSamples = maxSamples;
        _seed = seed;
    }

    /// <summary>
    /// Creates a new JSONL streaming loader for a single file.
    /// </summary>
    public JsonlStreamingLoader(
        string filePath,
        string? textField = null,
        string? labelField = null,
        int shuffleBufferSize = 0,
        int? maxSamples = null,
        int? seed = null)
        : this(new[] { filePath }, textField, labelField, shuffleBufferSize, maxSamples, seed)
    {
    }

    /// <summary>
    /// Reads raw JSON objects from the JSONL files.
    /// </summary>
    /// <returns>An enumerable of parsed JObject instances.</returns>
    public IEnumerable<JObject> ReadObjects()
    {
        int samplesRead = 0;

        if (_shuffleBufferSize > 1)
        {
            var random = _seed.HasValue
                ? RandomHelper.CreateSeededRandom(_seed.Value)
                : RandomHelper.CreateSecureRandom();
            var buffer = new List<JObject>();

            foreach (var obj in ReadObjectsRaw())
            {
                if (_maxSamples.HasValue && samplesRead >= _maxSamples.Value) yield break;

                buffer.Add(obj);
                if (buffer.Count >= _shuffleBufferSize)
                {
                    for (int i = buffer.Count - 1; i > 0; i--)
                    {
                        int j = random.Next(i + 1);
                        var temp = buffer[i];
                        buffer[i] = buffer[j];
                        buffer[j] = temp;
                    }
                    foreach (var item in buffer)
                    {
                        yield return item;
                        samplesRead++;
                        if (_maxSamples.HasValue && samplesRead >= _maxSamples.Value) yield break;
                    }
                    buffer.Clear();
                }
            }

            // Flush
            for (int i = buffer.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                var temp = buffer[i];
                buffer[i] = buffer[j];
                buffer[j] = temp;
            }
            foreach (var item in buffer)
            {
                if (_maxSamples.HasValue && samplesRead >= _maxSamples.Value) yield break;
                yield return item;
                samplesRead++;
            }
        }
        else
        {
            foreach (var obj in ReadObjectsRaw())
            {
                if (_maxSamples.HasValue && samplesRead >= _maxSamples.Value) yield break;
                yield return obj;
                samplesRead++;
            }
        }
    }

    /// <summary>
    /// Reads text and optional label pairs from the JSONL files.
    /// </summary>
    /// <returns>An enumerable of (text, label) pairs where label may be null if no label field is specified.</returns>
    public IEnumerable<(string Text, string? Label)> ReadTextSamples()
    {
        foreach (var obj in ReadObjects())
        {
            string text;
            if (_textField is not null)
            {
                var textToken = obj[_textField];
                text = textToken?.ToString() ?? string.Empty;
            }
            else
            {
                text = obj.ToString(Newtonsoft.Json.Formatting.None);
            }

            string? label = null;
            if (_labelField is not null)
            {
                var labelToken = obj[_labelField];
                label = labelToken?.ToString();
            }

            yield return (text, label);
        }
    }

    private IEnumerable<JObject> ReadObjectsRaw()
    {
        foreach (string filePath in _filePaths)
        {
            using var reader = new StreamReader(filePath, System.Text.Encoding.UTF8);
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                JObject? obj;
                try
                {
                    obj = JObject.Parse(line);
                }
                catch (Newtonsoft.Json.JsonException)
                {
                    continue; // Skip malformed lines
                }

                if (obj is not null)
                {
                    yield return obj;
                }
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}

using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Parses the <see href="https://github.com/huggingface/safetensors">safetensors</see> weight format — the
/// safe, simple format used to distribute pretrained model weights — into a <see cref="SafetensorsFile"/>.
/// This is the loading primitive for bringing external weights into AiDotNet's local engine.
/// </summary>
/// <remarks>
/// <para>
/// Layout: an 8-byte little-endian header length, a JSON header mapping tensor names to
/// <c>{dtype, shape, data_offsets}</c> (plus an optional <c>__metadata__</c>), then the concatenated tensor
/// bytes. This reader validates the framing and exposes each tensor; mapping the tensors to a particular
/// network's parameters (which depends on the architecture's layer naming) is a separate step.
/// </para>
/// <para><b>For Beginners:</b> Hand it a <c>.safetensors</c> file (bytes or a stream) and it tells you every
/// weight array inside and lets you read them — the first step in loading a downloaded model.
/// </para>
/// </remarks>
public static class SafetensorsReader
{
    /// <summary>
    /// Parses a safetensors file from a byte array.
    /// </summary>
    /// <param name="data">The full file contents.</param>
    /// <returns>The parsed file.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="data"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidDataException">Thrown when the framing or header is malformed.</exception>
    public static SafetensorsFile Read(byte[] data)
    {
        Guard.NotNull(data);
        if (data.Length < 8)
        {
            throw new InvalidDataException("safetensors data is too short to contain a header length.");
        }

        var headerLength = (long)BitConverter.ToUInt64(data, 0);
        if (headerLength < 0 || 8 + headerLength > data.Length)
        {
            throw new InvalidDataException("safetensors header length exceeds the data size.");
        }

        var headerJson = Encoding.UTF8.GetString(data, 8, checked((int)headerLength));
        JObject header;
        try
        {
            header = JObject.Parse(headerJson);
        }
        catch (Newtonsoft.Json.JsonException ex)
        {
            throw new InvalidDataException("safetensors header is not valid JSON: " + ex.Message);
        }

        var dataStart = 8 + headerLength;
        var tensors = new List<SafetensorsTensor>();
        foreach (var property in header)
        {
            if (property.Key == "__metadata__" || property.Value is not JObject entry)
            {
                continue;
            }

            var dtype = (string?)entry["dtype"] ?? throw new InvalidDataException($"Tensor '{property.Key}' is missing 'dtype'.");
            var shape = (entry["shape"] as JArray)?.Select(t => (long)t).ToList() ?? new List<long>();
            if (entry["data_offsets"] is not JArray offsets || offsets.Count != 2)
            {
                throw new InvalidDataException($"Tensor '{property.Key}' is missing valid 'data_offsets'.");
            }

            var begin = (long)offsets[0];
            var end = (long)offsets[1];
            if (begin < 0 || end < begin || dataStart + end > data.Length)
            {
                throw new InvalidDataException($"Tensor '{property.Key}' has out-of-range data offsets.");
            }

            tensors.Add(new SafetensorsTensor(property.Key, dtype, shape, begin, end));
        }

        return new SafetensorsFile(data, dataStart, tensors);
    }

    /// <summary>
    /// Parses a safetensors file from a stream (fully read into memory).
    /// </summary>
    /// <param name="stream">The stream to read.</param>
    /// <returns>The parsed file.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="stream"/> is <c>null</c>.</exception>
    public static SafetensorsFile Read(Stream stream)
    {
        Guard.NotNull(stream);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        return Read(memory.ToArray());
    }
}

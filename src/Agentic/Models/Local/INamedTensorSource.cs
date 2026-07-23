namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A source of named weight tensors readable as <see cref="double"/> arrays — the common surface over the
/// safetensors and GGUF readers that <see cref="WeightImporter"/> imports from.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Both file formats (safetensors, GGUF) can list their weight arrays by name and
/// hand them back as numbers. This interface is that shared ability, so the importer doesn't care which format
/// the weights came from.
/// </para>
/// </remarks>
public interface INamedTensorSource
{
    /// <summary>Gets the names of the tensors available in this source.</summary>
    IReadOnlyCollection<string> TensorNames { get; }

    /// <summary>Reads a tensor's values as <see cref="double"/>.</summary>
    /// <param name="name">The tensor name.</param>
    /// <returns>The tensor values in stored order.</returns>
    double[] ReadAsDouble(string name);
}

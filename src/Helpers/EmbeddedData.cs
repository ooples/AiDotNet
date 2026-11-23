using System.IO;
using System.Reflection;
using AiDotNet.Resources;

namespace AiDotNet.Helpers;

/// <summary>
/// Access to the tokenizer vocabularies and text-to-speech dictionaries embedded in the
/// AiDotNet.Resources assembly.
/// </summary>
/// <remarks>
/// The data is not embedded in this assembly because this assembly is built once per target
/// framework, so the package would carry a copy of every file for each of them. Build and license
/// material (see <see cref="BuildKeyProvider"/>) is still embedded here and is not served by this type.
/// </remarks>
internal static class EmbeddedData
{
    /// <summary>The assembly that holds the embedded data files.</summary>
    internal static Assembly Assembly { get; } = typeof(ResourceAnchor).Assembly;

    /// <summary>
    /// Opens an embedded data file by its logical name, for example
    /// <c>AiDotNet.Tokenization.t5_spiece.model.gz</c>.
    /// </summary>
    /// <param name="logicalName">The resource's logical name.</param>
    /// <returns>The resource stream, or <c>null</c> when no resource has that name.</returns>
    internal static Stream? Open(string logicalName) => Assembly.GetManifestResourceStream(logicalName);
}

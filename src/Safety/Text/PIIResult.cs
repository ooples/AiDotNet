namespace AiDotNet.Safety.Text;

/// <summary>
/// Detailed result from PII detection with detected entities and redacted text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> PIIResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class PIIResult
{
    /// <summary>Whether any PII was detected.</summary>
    public bool ContainsPII { get; init; }

    /// <summary>List of detected PII entities with their locations, types, and confidence.</summary>
    public IReadOnlyList<PIIEntity> Entities { get; init; } = Array.Empty<PIIEntity>();

    /// <summary>The text with PII redacted according to the configured strategy.</summary>
    public string RedactedText { get; init; } = string.Empty;

    /// <summary>Count of PII entities by type.</summary>
    public IReadOnlyDictionary<string, int> EntityCounts { get; init; } = new Dictionary<string, int>();
}

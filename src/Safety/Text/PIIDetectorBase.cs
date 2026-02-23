namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for PII detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for PII detectors including entity deduplication
/// and common regex timeout configuration. Concrete implementations provide the actual
/// detection strategy (regex patterns, NER, context-aware, or composite).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all PII detectors.
/// Each detector type extends this class and adds its own way of finding personal
/// information in text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class PIIDetectorBase<T> : TextSafetyModuleBase<T>, IPIIDetector<T>
{
    /// <summary>
    /// Maximum time in milliseconds to allow for regex operations (ReDoS protection).
    /// </summary>
    protected const int RegexTimeoutMs = 100;

    /// <inheritdoc />
    public abstract IReadOnlyList<PIIEntity> DetectPII(string text);

    /// <summary>
    /// Deduplicates overlapping PII entities, keeping the one with highest confidence.
    /// </summary>
    protected static IReadOnlyList<PIIEntity> DeduplicateEntities(List<PIIEntity> entities)
    {
        if (entities.Count <= 1) return entities;

        entities.Sort((a, b) => a.StartIndex.CompareTo(b.StartIndex));
        var result = new List<PIIEntity> { entities[0] };

        for (int i = 1; i < entities.Count; i++)
        {
            var last = result[result.Count - 1];
            if (entities[i].StartIndex < last.EndIndex)
            {
                // Overlapping — keep the higher confidence one
                if (entities[i].Confidence > last.Confidence)
                {
                    result[result.Count - 1] = entities[i];
                }
            }
            else
            {
                result.Add(entities[i]);
            }
        }

        return result;
    }
}

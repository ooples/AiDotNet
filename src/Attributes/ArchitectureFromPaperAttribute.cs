namespace AiDotNet.Attributes;

/// <summary>
/// Declares that a model's layers come from a factory built for another paper's architecture, and names
/// that paper.
/// </summary>
/// <remarks>
/// <para>
/// A shared layer factory is legitimate when the models really share an architecture (Whisper variants
/// are Whisper). It is a defect when a generic template stands in for a paper's own design, as the OCR
/// template models did. The analyzer cannot tell the two apart, so a model whose factory is also used by
/// models of other papers must say which architecture it is reusing (ADNDEF003). The declaration is a
/// reviewed claim: it names the paper the reused architecture belongs to.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ArchitectureFromPaperAttribute : Attribute
{
    /// <summary>The URL of the paper whose architecture the shared factory implements.</summary>
    public string PaperUrl { get; }

    /// <summary>Why reusing that architecture is faithful to this model's own paper.</summary>
    public string Reason { get; }

    /// <summary>Creates the declaration.</summary>
    public ArchitectureFromPaperAttribute(string paperUrl, string reason)
    {
        PaperUrl = paperUrl ?? throw new ArgumentNullException(nameof(paperUrl));
        Reason = reason ?? throw new ArgumentNullException(nameof(reason));
    }
}
namespace AiDotNet.Enums;

/// <summary>
/// Strategy for redacting detected PII from text.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When personal information is found in text, you can choose
/// how to handle it. Masking replaces it with asterisks, hashing replaces it with a
/// consistent hash, replacement uses a placeholder, and removal deletes it entirely.
/// </para>
/// </remarks>
public enum RedactionStrategy
{
    /// <summary>Replace PII with asterisks (e.g., "John" → "****").</summary>
    Mask,

    /// <summary>Replace PII with a consistent hash (e.g., "John" → "[HASH:a1b2c3]").</summary>
    Hash,

    /// <summary>Replace PII with a type placeholder (e.g., "John" → "[PERSON]").</summary>
    Replace,

    /// <summary>Remove PII entirely from the text.</summary>
    Remove,

    /// <summary>Replace PII with a reversible token for later de-anonymization.</summary>
    Tokenize
}

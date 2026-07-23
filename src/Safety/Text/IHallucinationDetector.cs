using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Interface for hallucination detection modules that identify fabricated or unfaithful content.
/// </summary>
/// <remarks>
/// <para>
/// Hallucination detectors analyze model outputs to identify claims that are not grounded
/// in source material or are factually inconsistent. Approaches include reference-based
/// comparison, self-consistency checking, knowledge triplet extraction, and NLI entailment.
/// </para>
/// <para>
/// <b>For Beginners:</b> A hallucination detector checks if an AI made something up.
/// It can compare the AI's output against source documents, check if the AI contradicts
/// itself, or verify specific facts. This helps ensure AI outputs are trustworthy.
/// </para>
/// <para>
/// <b>References:</b>
/// - RefChecker: Knowledge triplet-based detection (Amazon, 2024, arxiv:2405.14486)
/// - HHEM 2.1/2.3: Production-grade detection (Vectara, 2024-2025)
/// - ReDeEP: Hallucination detection in RAG systems (ICLR 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IHallucinationDetector<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Gets the hallucination likelihood score (0.0 = grounded, 1.0 = fabricated).
    /// </summary>
    /// <param name="text">The text to evaluate for hallucination.</param>
    /// <returns>A hallucination probability score between 0.0 and 1.0.</returns>
    double GetHallucinationScore(string text);

    /// <summary>
    /// Evaluates text against reference content for faithfulness.
    /// </summary>
    /// <param name="generatedText">The generated text to check.</param>
    /// <param name="referenceText">The reference/source text to check against.</param>
    /// <returns>A list of safety findings for any detected hallucinations.</returns>
    IReadOnlyList<SafetyFinding> EvaluateAgainstReference(string generatedText, string referenceText);
}

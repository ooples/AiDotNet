using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Interface for toxicity detection modules that identify harmful, abusive, or toxic text content.
/// </summary>
/// <remarks>
/// <para>
/// Toxicity detectors analyze text for hate speech, harassment, threats, profanity, and other
/// forms of harmful language. Implementations range from rule-based pattern matching to
/// embedding similarity and trained classifier approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> A toxicity detector checks if text contains harmful language like
/// insults, threats, or hate speech. Different implementations use different techniques —
/// some match known bad words (rule-based), others understand meaning (embedding-based),
/// and others use trained models (classifier-based). The ensemble combines all of these.
/// </para>
/// <para>
/// <b>References:</b>
/// - MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
/// - LLM-extracted rationales for interpretable hate speech detection (2024, arxiv:2403.12403)
/// - GPT-4o/LLaMA-3 zero-shot hate speech detection (2025, arxiv:2506.12744)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ITextToxicityDetector<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Gets the toxicity score for the given text (0.0 = safe, 1.0 = maximally toxic).
    /// </summary>
    /// <param name="text">The text to score.</param>
    /// <returns>A toxicity score between 0.0 and 1.0.</returns>
    double GetToxicityScore(string text);
}

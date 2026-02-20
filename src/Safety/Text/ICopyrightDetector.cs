using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Interface for copyright and memorization detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Copyright detectors identify potential copyright violations and training data memorization
/// in model outputs. They use n-gram overlap analysis, embedding similarity to known works,
/// and perplexity-based memorization detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> A copyright detector checks if an AI's output copies from
/// copyrighted books, articles, or code. It can detect when the AI is regurgitating
/// memorized training data rather than generating original content.
/// </para>
/// <para>
/// <b>References:</b>
/// - DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
/// - Machine unlearning to remove memorized copyrighted content (2024, arxiv:2412.18621)
/// - GPTZero: Hierarchical multi-task AI text detection (2026, arxiv:2602.13042)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ICopyrightDetector<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Gets the memorization likelihood score (0.0 = original, 1.0 = memorized).
    /// </summary>
    /// <param name="text">The text to evaluate.</param>
    /// <returns>A memorization probability score between 0.0 and 1.0.</returns>
    double GetMemorizationScore(string text);
}

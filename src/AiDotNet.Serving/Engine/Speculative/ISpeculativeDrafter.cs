using System.Collections.Generic;

namespace AiDotNet.Serving.Engine.Speculative;

/// <summary>
/// Proposes candidate continuation tokens ("draft") that the target model then verifies in a single forward
/// pass. A good drafter is cheap and often right; because verification always corrects it, a drafter can never
/// change the generated output — only the speed. This is the pluggable draft source of speculative decoding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> speculative decoding speeds up generation by <i>guessing</i> several next tokens
/// with something cheap, then letting the real (expensive) model check all the guesses at once. This interface
/// is the guesser. If the guesses are good, the model confirms many tokens in one step instead of one; if they
/// are wrong, the model's own answer is used instead — so the result is always exactly what the model would
/// have produced anyway.</para>
/// </remarks>
public interface ISpeculativeDrafter
{
    /// <summary>
    /// Proposes up to <paramref name="maxDraftTokens"/> continuation tokens for the given context. May return
    /// fewer (including none) when the drafter has no confident guess.
    /// </summary>
    IReadOnlyList<int> Draft(IReadOnlyList<int> contextTokenIds, int maxDraftTokens);
}

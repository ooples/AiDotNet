using AiDotNet.VisionLanguage.Grounding;

namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// A model that trains on grounded conversations: an image, a conversation about it, the regions its questions name, and
/// a mask for every <c>[SEG]</c> in its answers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>Such a model's loss combines the language-model loss with mask losses; a tensor target cannot carry the
/// conversation, so these models refuse <c>Train(input, expected)</c> and train here instead.</remarks>
public interface IGroundedConversationTrainable<T>
{
    /// <summary>One training step with the model's paper loss.</summary>
    void TrainConversation(GlammSample<T> sample);

    /// <summary>The paper loss on the same sample, without training.</summary>
    T EvaluateConversationLoss(GlammSample<T> sample);
}

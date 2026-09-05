namespace AiDotNet.Enums;

/// <summary>Names the stage of the novelty pipeline that settled a candidate's fate.</summary>
/// <remarks>
/// <para>
/// The pipeline is a cost ladder: a structural distance costs arithmetic over text already in memory, an embedding
/// comparison costs a provider request unless the vector is cached, and a language-model judgement costs a full
/// completion. Recording which rung decided lets a run report what its novelty gate actually spent instead of
/// guessing, and it is the evidence behind any claim that most decisions never left the first rung.
/// </para>
/// <para><b>For Beginners:</b> Deciding whether a new candidate is "really new" can be done cheaply or expensively.
/// This value tells you which method produced the answer for one candidate, so you can see how often the expensive
/// methods were needed at all.</para>
/// </remarks>
public enum ProgramNoveltyStage
{
    /// <summary>Nothing was compared, because there was nothing to compare against.</summary>
    None = 0,

    /// <summary>A structural distance decided; no embedding request and no model call were made.</summary>
    Structural = 1,

    /// <summary>An embedding cosine comparison decided.</summary>
    Embedding = 2,

    /// <summary>A language-model judgement decided.</summary>
    LanguageModel = 3
}

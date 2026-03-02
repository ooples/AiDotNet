namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// SuperGLUE benchmark sub-tasks.
/// </summary>
public enum SuperGlueTask
{
    /// <summary>Boolean Questions (passage + question, binary).</summary>
    BoolQ,
    /// <summary>CommitmentBank (premise + hypothesis, 3-class).</summary>
    CB,
    /// <summary>Choice of Plausible Alternatives (sentence + 2 choices).</summary>
    COPA,
    /// <summary>Multi-Sentence Reading Comprehension (paragraph + question + answer, binary).</summary>
    MultiRC,
    /// <summary>Reading Comprehension with Commonsense Reasoning (passage + query, entity extraction).</summary>
    ReCoRD,
    /// <summary>Recognizing Textual Entailment (premise + hypothesis, binary).</summary>
    RTE,
    /// <summary>Words in Context (two sentences + word, binary).</summary>
    WiC,
    /// <summary>Winograd Schema Challenge (sentence + pronoun, binary).</summary>
    WSC
}

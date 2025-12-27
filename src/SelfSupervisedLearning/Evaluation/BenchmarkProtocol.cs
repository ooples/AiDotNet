namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Supported evaluation protocols for transfer learning benchmarks.
/// </summary>
public enum BenchmarkProtocol
{
    /// <summary>Freeze encoder, train linear classifier only.</summary>
    LinearProbing,
    /// <summary>Fine-tune entire network with lower learning rate.</summary>
    FineTuning,
    /// <summary>Few-shot learning with limited labeled data.</summary>
    FewShot
}

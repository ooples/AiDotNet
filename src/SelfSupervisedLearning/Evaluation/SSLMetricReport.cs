namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Complete SSL metrics report.
/// </summary>
public class SSLMetricReport<T>
{
    /// <summary>Standard deviation of first view representations.</summary>
    public T Std1 { get; set; } = default!;

    /// <summary>Standard deviation of second view representations.</summary>
    public T Std2 { get; set; } = default!;

    /// <summary>Alignment between positive pairs.</summary>
    public T Alignment { get; set; } = default!;

    /// <summary>Uniformity of first view representations.</summary>
    public T Uniformity1 { get; set; } = default!;

    /// <summary>Uniformity of second view representations.</summary>
    public T Uniformity2 { get; set; } = default!;

    /// <summary>Effective rank of first view representations.</summary>
    public T EffectiveRank1 { get; set; } = default!;

    /// <summary>Effective rank of second view representations.</summary>
    public T EffectiveRank2 { get; set; } = default!;

    /// <summary>Whether representation collapse was detected.</summary>
    public bool CollapseDetected { get; set; }
}

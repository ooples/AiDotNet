namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Complete SSL metrics report.
/// </summary>
public class SSLMetricReport<T>
{
    /// <summary>Standard deviation of first view representations.</summary>
    public required T Std1 { get; set; }

    /// <summary>Standard deviation of second view representations.</summary>
    public required T Std2 { get; set; }

    /// <summary>Alignment between positive pairs.</summary>
    public required T Alignment { get; set; }

    /// <summary>Uniformity of first view representations.</summary>
    public required T Uniformity1 { get; set; }

    /// <summary>Uniformity of second view representations.</summary>
    public required T Uniformity2 { get; set; }

    /// <summary>Effective rank of first view representations.</summary>
    public required T EffectiveRank1 { get; set; }

    /// <summary>Effective rank of second view representations.</summary>
    public required T EffectiveRank2 { get; set; }

    /// <summary>Whether representation collapse was detected.</summary>
    public bool CollapseDetected { get; set; }
}

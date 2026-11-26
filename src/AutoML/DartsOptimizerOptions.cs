namespace AiDotNet.AutoML;

/// <summary>
/// Optimizer settings of the DARTS search in <see cref="NeuralArchitectureSearch{T}"/>. The defaults are those
/// of Liu et al. 2019, "DARTS: Differentiable Architecture Search" (Sec. 5.1).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DARTS trains the architecture weights (alpha) with Adam and the network weights
/// with SGD. The defaults reproduce the CIFAR-10 search of the paper; a different search space or dataset may
/// need, for example, a different network-weight learning rate.</para>
/// </remarks>
public class DartsOptimizerOptions
{
    /// <summary>Adam learning rate of the architecture parameters (default: 3e-4).</summary>
    public double ArchitectureLearningRate { get; set; } = 3e-4;

    /// <summary>Adam beta1 of the architecture parameters (default: 0.5).</summary>
    public double ArchitectureBeta1 { get; set; } = 0.5;

    /// <summary>Adam beta2 of the architecture parameters (default: 0.999).</summary>
    public double ArchitectureBeta2 { get; set; } = 0.999;

    /// <summary>Adam epsilon of the architecture parameters (default: 1e-8).</summary>
    public double ArchitectureEpsilon { get; set; } = 1e-8;

    /// <summary>Weight decay of the architecture parameters (default: 1e-3).</summary>
    public double ArchitectureWeightDecay { get; set; } = 1e-3;

    /// <summary>Initial (maximum) learning rate of the cosine schedule for the network weights (default: 0.025).</summary>
    public double WeightMaxLearningRate { get; set; } = 0.025;

    /// <summary>Final (minimum) learning rate of the cosine schedule for the network weights (default: 0.001).</summary>
    public double WeightMinLearningRate { get; set; } = 0.001;

    /// <summary>SGD momentum of the network weights (default: 0.9).</summary>
    public double WeightMomentum { get; set; } = 0.9;

    /// <summary>Weight decay of the network weights (default: 3e-4).</summary>
    public double WeightDecay { get; set; } = 3e-4;

    /// <summary>Global gradient-norm clip of the network weights (default: 5).</summary>
    public double GradientClipNorm { get; set; } = 5.0;
}

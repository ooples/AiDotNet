namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for robustness evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Robustness evaluation measures how well a model performs under input perturbations,
/// noise, and adversarial attacks. A robust model maintains performance despite variations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Robustness tests answer questions like:
/// <list type="bullet">
/// <item>What happens if there's noise in the input? (Noise robustness)</item>
/// <item>What if someone intentionally tries to fool the model? (Adversarial robustness)</item>
/// <item>What if the input has small errors or missing values? (Perturbation robustness)</item>
/// </list>
/// This is important for real-world deployments where inputs aren't always clean.
/// </para>
/// </remarks>
public class RobustnessOptions
{
    /// <summary>
    /// Whether to test Gaussian noise robustness. Default: true.
    /// </summary>
    public bool? TestGaussianNoise { get; set; }

    /// <summary>
    /// Noise levels (standard deviations) to test. Default: [0.01, 0.05, 0.1, 0.2].
    /// </summary>
    public double[]? NoiseLevels { get; set; }

    /// <summary>
    /// Number of noise samples per input. Default: 10.
    /// </summary>
    public int? NoiseSamplesPerInput { get; set; }

    /// <summary>
    /// Whether to test uniform noise. Default: false.
    /// </summary>
    public bool? TestUniformNoise { get; set; }

    /// <summary>
    /// Whether to test feature perturbation impact. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tests how much each feature affects predictions
    /// when slightly changed. High sensitivity to a feature might indicate fragility.</para>
    /// </remarks>
    public bool? TestFeaturePerturbation { get; set; }

    /// <summary>
    /// Perturbation magnitude (relative). Default: 0.1 (10%).
    /// </summary>
    public double? PerturbationMagnitude { get; set; }

    /// <summary>
    /// Specific features to perturb. Default: null (all features).
    /// </summary>
    public int[]? FeaturesToPerturb { get; set; }

    /// <summary>
    /// Whether to test adversarial robustness. Default: false (expensive).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adversarial attacks find the smallest change to an input
    /// that flips the prediction. This tests worst-case robustness.</para>
    /// </remarks>
    public bool? TestAdversarialRobustness { get; set; }

    /// <summary>
    /// Adversarial attack method. Default: FGSM.
    /// </summary>
    public AdversarialAttackMethod? AttackMethod { get; set; }

    /// <summary>
    /// Epsilon for adversarial attacks. Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon limits how much the adversarial example can
    /// differ from the original. Smaller epsilon = harder attack to defend against.</para>
    /// </remarks>
    public double? AdversarialEpsilon { get; set; }

    /// <summary>
    /// Number of attack iterations (for iterative methods). Default: 10.
    /// </summary>
    public int? AttackIterations { get; set; }

    /// <summary>
    /// Whether to test missing value robustness. Default: false.
    /// </summary>
    public bool? TestMissingValues { get; set; }

    /// <summary>
    /// Missing value rates to test. Default: [0.05, 0.1, 0.2].
    /// </summary>
    public double[]? MissingValueRates { get; set; }

    /// <summary>
    /// Feature dropout rates to test (random feature removal). Default: [0.1, 0.2, 0.3, 0.5].
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tests how the model performs when random features
    /// are set to zero, simulating missing data or feature failures.</para>
    /// </remarks>
    public double[]? DropoutRates { get; set; }

    /// <summary>
    /// Missing value handling strategy. Default: MeanImputation.
    /// </summary>
    public MissingValueStrategy? MissingStrategy { get; set; }

    /// <summary>
    /// Whether to test outlier robustness. Default: false.
    /// </summary>
    public bool? TestOutlierRobustness { get; set; }

    /// <summary>
    /// Outlier contamination rates to test. Default: [0.05, 0.1].
    /// </summary>
    public double[]? OutlierRates { get; set; }

    /// <summary>
    /// Whether to compute input gradient norms. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Large input gradients indicate sensitivity to small
    /// input changes. Smaller gradients generally mean more robust predictions.</para>
    /// </remarks>
    public bool? ComputeInputGradients { get; set; }

    /// <summary>
    /// Whether to estimate local Lipschitz constant. Default: false (expensive).
    /// </summary>
    public bool? EstimateLipschitzConstant { get; set; }

    /// <summary>
    /// Number of samples for Lipschitz estimation. Default: 100.
    /// </summary>
    public int? LipschitzSamples { get; set; }

    /// <summary>
    /// Whether to compute prediction flip rate under perturbation. Default: true.
    /// </summary>
    public bool? ComputeFlipRate { get; set; }

    /// <summary>
    /// Whether to run tests in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }

    /// <summary>
    /// Maximum parallelism. Default: null (all cores).
    /// </summary>
    public int? MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Subset of samples to test (ratio). Default: 1.0 (all).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to less than 1.0 to test on a subset of data,
    /// which is faster but less comprehensive.</para>
    /// </remarks>
    public double? SampleRatio { get; set; }

    /// <summary>
    /// Maximum samples to test. Default: null (no limit).
    /// </summary>
    public int? MaxSamples { get; set; }
}

/// <summary>
/// Methods for generating adversarial examples.
/// </summary>
public enum AdversarialAttackMethod
{
    /// <summary>
    /// Fast Gradient Sign Method - simple and fast.
    /// </summary>
    FGSM = 0,

    /// <summary>
    /// Projected Gradient Descent - iterative, stronger.
    /// </summary>
    PGD = 1,

    /// <summary>
    /// Basic Iterative Method - iterative FGSM.
    /// </summary>
    BIM = 2,

    /// <summary>
    /// Carlini & Wagner L2 attack - strongest but slowest.
    /// </summary>
    CW = 3,

    /// <summary>
    /// DeepFool - finds minimal perturbation.
    /// </summary>
    DeepFool = 4,

    /// <summary>
    /// AutoAttack - ensemble of attacks.
    /// </summary>
    AutoAttack = 5
}

/// <summary>
/// Strategies for handling missing values in robustness testing.
/// </summary>
public enum MissingValueStrategy
{
    /// <summary>Replace with feature mean.</summary>
    MeanImputation = 0,

    /// <summary>Replace with feature median.</summary>
    MedianImputation = 1,

    /// <summary>Replace with zero.</summary>
    ZeroImputation = 2,

    /// <summary>Replace with feature mode (most common value).</summary>
    ModeImputation = 3,

    /// <summary>Use model's native missing value handling.</summary>
    ModelNative = 4
}

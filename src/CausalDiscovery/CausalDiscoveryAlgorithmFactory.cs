using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Factory for creating causal discovery algorithm instances from enum types.
/// </summary>
/// <remarks>
/// <para>
/// This factory maps <see cref="CausalDiscoveryAlgorithmType"/> enum values to their
/// corresponding algorithm implementations. It supports all 72 algorithms in the framework.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this factory when you want to create an algorithm by name
/// (e.g., from configuration) rather than directly instantiating a class. Just pass the
/// algorithm type and optional settings, and the factory creates the right algorithm.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class CausalDiscoveryAlgorithmFactory<T>
{
    /// <summary>
    /// Creates an algorithm instance for the specified type.
    /// </summary>
    /// <param name="algorithmType">The algorithm to create.</param>
    /// <param name="options">Optional configuration for the algorithm.</param>
    /// <returns>An instance of the requested causal discovery algorithm.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an unknown algorithm type is specified.</exception>
    public static ICausalDiscoveryAlgorithm<T> Create(
        CausalDiscoveryAlgorithmType algorithmType,
        CausalDiscoveryOptions? options = null)
    {
        return algorithmType switch
        {
            // Category 1: Continuous Optimization
            CausalDiscoveryAlgorithmType.NOTEARSLinear => new ContinuousOptimization.NOTEARSLinear<T>(options),
            CausalDiscoveryAlgorithmType.NOTEARSNonlinear => new ContinuousOptimization.NOTEARSNonlinear<T>(options),
            CausalDiscoveryAlgorithmType.NOTEARSSobolev => new ContinuousOptimization.NOTEARSSobolev<T>(options),
            CausalDiscoveryAlgorithmType.NOTEARSLowRank => new ContinuousOptimization.NOTEARSLowRank<T>(options),
            CausalDiscoveryAlgorithmType.DAGMALinear => new ContinuousOptimization.DAGMALinear<T>(options),
            CausalDiscoveryAlgorithmType.DAGMANonlinear => new ContinuousOptimization.DAGMANonlinear<T>(options),
            CausalDiscoveryAlgorithmType.GOLEM => new ContinuousOptimization.GOLEMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.NoCurl => new ContinuousOptimization.NoCurlAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.MCSL => new ContinuousOptimization.MCSLAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CORL => new ContinuousOptimization.CORLAlgorithm<T>(options),

            // Category 2: Score-Based Search
            CausalDiscoveryAlgorithmType.GES => new ScoreBased.GESAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.FGES => new ScoreBased.FGESAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.HillClimbing => new ScoreBased.HillClimbingAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.TabuSearch => new ScoreBased.TabuSearchAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.K2 => new ScoreBased.K2Algorithm<T>(options),
            CausalDiscoveryAlgorithmType.GRaSP => new ScoreBased.GRaSPAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.BOSS => new ScoreBased.BOSSAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.ExactSearch => new ScoreBased.ExactSearchAlgorithm<T>(options),

            // Category 3: Constraint-Based
            CausalDiscoveryAlgorithmType.PC => new ConstraintBased.PCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.FCI => new ConstraintBased.FCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.RFCI => new ConstraintBased.RFCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.MMPC => new ConstraintBased.MMPCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CPC => new ConstraintBased.CPCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CDNOD => new ConstraintBased.CDNODAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.IAMB => new ConstraintBased.IAMBAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.FastIAMB => new ConstraintBased.FastIAMBAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.MarkovBlanket => new ConstraintBased.MarkovBlanketAlgorithm<T>(options),

            // Category 4: Hybrid
            CausalDiscoveryAlgorithmType.MMHC => new Hybrid.MMHCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.H2PC => new Hybrid.H2PCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.GFCI => new Hybrid.GFCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.PCNOTEARS => new Hybrid.PCNOTEARSAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.RSMAX2 => new Hybrid.RSMAX2Algorithm<T>(options),

            // Category 5: Functional / ICA-Based
            CausalDiscoveryAlgorithmType.ICALiNGAM => new Functional.ICALiNGAMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.DirectLiNGAM => new Functional.DirectLiNGAMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.VARLiNGAM => new Functional.VARLiNGAMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.RCD => new Functional.RCDAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CAMUV => new Functional.CAMUVAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.ANM => new Functional.ANMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.PNL => new Functional.PNLAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.IGCI => new Functional.IGCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CAM => new Functional.CAMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CCDr => new Functional.CCDrAlgorithm<T>(options),

            // Category 6: Time Series
            CausalDiscoveryAlgorithmType.GrangerCausality => new TimeSeries.GrangerCausalityAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.PCMCI => new TimeSeries.PCMCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.PCMCIPlus => new TimeSeries.PCMCIPlusAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.DYNOTEARS => new TimeSeries.DYNOTEARSAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.TiMINo => new TimeSeries.TiMINoAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.TSFCI => new TimeSeries.TSFCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.LPCMCI => new TimeSeries.LPCMCIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.NTSNOTEARS => new TimeSeries.NTSNOTEARSAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CCM => new TimeSeries.CCMAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.NeuralGranger => new TimeSeries.NeuralGrangerAlgorithm<T>(options),

            // Category 7: Deep Learning
            CausalDiscoveryAlgorithmType.DAGGNN => new DeepLearning.DAGGNNAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.GraNDAG => new DeepLearning.GraNDAGAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CASTLE => new DeepLearning.CASTLEAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.DECI => new DeepLearning.DECIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.GAE => new DeepLearning.GAEAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CGNN => new DeepLearning.CGNNAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.TCDF => new DeepLearning.TCDFAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.AmortizedCD => new DeepLearning.AmortizedCDAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.AVICI => new DeepLearning.AVICIAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.CausalVAE => new DeepLearning.CausalVAEAlgorithm<T>(options),

            // Category 8: Bayesian
            CausalDiscoveryAlgorithmType.OrderMCMC => new Bayesian.OrderMCMCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.DiBS => new Bayesian.DiBSAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.BCDNets => new Bayesian.BCDNetsAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.BayesDAG => new Bayesian.BayesDAGAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.PartitionMCMC => new Bayesian.PartitionMCMCAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.IterativeMCMC => new Bayesian.IterativeMCMCAlgorithm<T>(options),

            // Category 9: Information-Theoretic
            CausalDiscoveryAlgorithmType.OCSE => new InformationTheoretic.OCSEAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.TransferEntropy => new InformationTheoretic.TransferEntropyAlgorithm<T>(options),
            CausalDiscoveryAlgorithmType.KraskovMI => new InformationTheoretic.KraskovMIAlgorithm<T>(options),

            // Category 10: Specialized
            CausalDiscoveryAlgorithmType.GOBNILP => new Specialized.GOBNILPAlgorithm<T>(options),

            _ => throw new ArgumentOutOfRangeException(nameof(algorithmType),
                $"Unknown causal discovery algorithm: {algorithmType}")
        };
    }

    /// <summary>
    /// Creates the default (recommended) algorithm for general-purpose causal discovery.
    /// </summary>
    /// <param name="options">Optional configuration.</param>
    /// <returns>A NOTEARS Linear algorithm instance (reliable, general-purpose default).</returns>
    public static ICausalDiscoveryAlgorithm<T> CreateDefault(CausalDiscoveryOptions? options = null)
        => Create(CausalDiscoveryAlgorithmType.NOTEARSLinear, options);

    /// <summary>
    /// Gets all available algorithm types as an enumerable.
    /// </summary>
    public static IEnumerable<CausalDiscoveryAlgorithmType> GetAvailableAlgorithms()
        => (CausalDiscoveryAlgorithmType[])Enum.GetValues(typeof(CausalDiscoveryAlgorithmType));

    /// <summary>
    /// Gets the category for a given algorithm type without instantiation.
    /// </summary>
    internal static CausalDiscoveryCategory GetCategory(CausalDiscoveryAlgorithmType algorithmType)
    {
        return algorithmType switch
        {
            CausalDiscoveryAlgorithmType.NOTEARSLinear or CausalDiscoveryAlgorithmType.NOTEARSNonlinear or
            CausalDiscoveryAlgorithmType.NOTEARSSobolev or CausalDiscoveryAlgorithmType.NOTEARSLowRank or
            CausalDiscoveryAlgorithmType.DAGMALinear or CausalDiscoveryAlgorithmType.DAGMANonlinear or
            CausalDiscoveryAlgorithmType.GOLEM or CausalDiscoveryAlgorithmType.NoCurl or
            CausalDiscoveryAlgorithmType.MCSL or CausalDiscoveryAlgorithmType.CORL
                => CausalDiscoveryCategory.ContinuousOptimization,

            CausalDiscoveryAlgorithmType.GES or CausalDiscoveryAlgorithmType.FGES or
            CausalDiscoveryAlgorithmType.HillClimbing or CausalDiscoveryAlgorithmType.TabuSearch or
            CausalDiscoveryAlgorithmType.K2 or CausalDiscoveryAlgorithmType.GRaSP or
            CausalDiscoveryAlgorithmType.BOSS or CausalDiscoveryAlgorithmType.ExactSearch
                => CausalDiscoveryCategory.ScoreBasedSearch,

            CausalDiscoveryAlgorithmType.PC or CausalDiscoveryAlgorithmType.FCI or
            CausalDiscoveryAlgorithmType.RFCI or CausalDiscoveryAlgorithmType.MMPC or
            CausalDiscoveryAlgorithmType.CPC or CausalDiscoveryAlgorithmType.CDNOD or
            CausalDiscoveryAlgorithmType.IAMB or CausalDiscoveryAlgorithmType.FastIAMB or
            CausalDiscoveryAlgorithmType.MarkovBlanket
                => CausalDiscoveryCategory.ConstraintBased,

            CausalDiscoveryAlgorithmType.MMHC or CausalDiscoveryAlgorithmType.H2PC or
            CausalDiscoveryAlgorithmType.GFCI or CausalDiscoveryAlgorithmType.PCNOTEARS or
            CausalDiscoveryAlgorithmType.RSMAX2
                => CausalDiscoveryCategory.Hybrid,

            CausalDiscoveryAlgorithmType.ICALiNGAM or CausalDiscoveryAlgorithmType.DirectLiNGAM or
            CausalDiscoveryAlgorithmType.VARLiNGAM or CausalDiscoveryAlgorithmType.RCD or
            CausalDiscoveryAlgorithmType.CAMUV or CausalDiscoveryAlgorithmType.ANM or
            CausalDiscoveryAlgorithmType.PNL or CausalDiscoveryAlgorithmType.IGCI or
            CausalDiscoveryAlgorithmType.CAM or CausalDiscoveryAlgorithmType.CCDr
                => CausalDiscoveryCategory.Functional,

            CausalDiscoveryAlgorithmType.GrangerCausality or CausalDiscoveryAlgorithmType.PCMCI or
            CausalDiscoveryAlgorithmType.PCMCIPlus or CausalDiscoveryAlgorithmType.DYNOTEARS or
            CausalDiscoveryAlgorithmType.TiMINo or CausalDiscoveryAlgorithmType.TSFCI or
            CausalDiscoveryAlgorithmType.LPCMCI or CausalDiscoveryAlgorithmType.NTSNOTEARS or
            CausalDiscoveryAlgorithmType.CCM or CausalDiscoveryAlgorithmType.NeuralGranger
                => CausalDiscoveryCategory.TimeSeries,

            CausalDiscoveryAlgorithmType.DAGGNN or CausalDiscoveryAlgorithmType.GraNDAG or
            CausalDiscoveryAlgorithmType.CASTLE or CausalDiscoveryAlgorithmType.DECI or
            CausalDiscoveryAlgorithmType.GAE or CausalDiscoveryAlgorithmType.CGNN or
            CausalDiscoveryAlgorithmType.TCDF or CausalDiscoveryAlgorithmType.AmortizedCD or
            CausalDiscoveryAlgorithmType.AVICI or CausalDiscoveryAlgorithmType.CausalVAE
                => CausalDiscoveryCategory.DeepLearning,

            CausalDiscoveryAlgorithmType.OrderMCMC or CausalDiscoveryAlgorithmType.DiBS or
            CausalDiscoveryAlgorithmType.BCDNets or CausalDiscoveryAlgorithmType.BayesDAG or
            CausalDiscoveryAlgorithmType.PartitionMCMC or CausalDiscoveryAlgorithmType.IterativeMCMC
                => CausalDiscoveryCategory.Bayesian,

            CausalDiscoveryAlgorithmType.OCSE or CausalDiscoveryAlgorithmType.TransferEntropy or
            CausalDiscoveryAlgorithmType.KraskovMI
                => CausalDiscoveryCategory.InformationTheoretic,

            CausalDiscoveryAlgorithmType.GOBNILP
                => CausalDiscoveryCategory.Specialized,

            _ => throw new ArgumentOutOfRangeException(nameof(algorithmType))
        };
    }
}

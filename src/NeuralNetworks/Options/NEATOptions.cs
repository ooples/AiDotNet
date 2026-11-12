using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the NEAT neural network.
/// </summary>
/// <remarks>
/// Defaults are the parameters of Stanley &amp; Miikkulainen (2002), Section 4.1, where the paper gives
/// them. The weight perturbation power and cap, which the paper leaves open, follow NEAT-Python.
/// </remarks>
public class NEATOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the fraction of offspring produced by crossover. Default: <c>0.75</c> (the paper's
    /// "25% of offspring resulted from mutation without crossover").
    /// </summary>
    public double CrossoverRate { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the probability that a genome's connection weights are mutated. Default: <c>0.8</c>.
    /// </summary>
    public double WeightMutationRate { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the probability that a mutated weight is replaced by a new random value instead of
    /// being perturbed. Default: <c>0.1</c> (the paper's 90% perturbed / 10% new value).
    /// </summary>
    public double WeightReplaceRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the half-width of the uniform weight perturbation. Default: <c>0.5</c> (NEAT-Python's
    /// weight_mutate_power; the paper does not state one).
    /// </summary>
    public double WeightPerturbPower { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the bound on connection weight magnitude. Default: <c>8.0</c>.
    /// </summary>
    public double WeightCap { get; set; } = 8.0;

    /// <summary>
    /// Gets or sets the probability of an add-node mutation. Default: <c>0.03</c>.
    /// </summary>
    public double AddNodeRate { get; set; } = 0.03;

    /// <summary>
    /// Gets or sets the probability of an add-connection mutation. Default: <c>0.05</c>.
    /// </summary>
    public double AddConnectionRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets how many node pairs an add-connection mutation tries before giving up. Default: <c>20</c>.
    /// </summary>
    public int AddConnectionTries { get; set; } = 20;

    /// <summary>
    /// Gets or sets the chance that an inherited gene is disabled when it was disabled in either parent.
    /// Default: <c>0.75</c>.
    /// </summary>
    public double DisableInheritedRate { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the interspecies mating rate. Default: <c>0.001</c>.
    /// </summary>
    public double InterspeciesMatingRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets c1, the weight of excess genes in the compatibility distance. Default: <c>1.0</c>.
    /// </summary>
    public double ExcessCoefficient { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets c2, the weight of disjoint genes in the compatibility distance. Default: <c>1.0</c>.
    /// </summary>
    public double DisjointCoefficient { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets c3, the weight of the average weight difference of matching genes. Default: <c>0.4</c>.
    /// </summary>
    public double WeightDifferenceCoefficient { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the compatibility threshold delta_t that separates species. Default: <c>3.0</c>.
    /// </summary>
    public double CompatibilityThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the genome size below which N (the normalizer of E and D) is 1. Default: <c>20</c>.
    /// </summary>
    public int GenomeSizeNormalizationThreshold { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of generations without improvement after which a species may no longer
    /// reproduce. Default: <c>15</c>.
    /// </summary>
    public int SpeciesStagnationLimit { get; set; } = 15;

    /// <summary>
    /// Gets or sets the number of generations without population-wide improvement after which only the
    /// top two species reproduce. Default: <c>20</c>.
    /// </summary>
    public int PopulationStagnationLimit { get; set; } = 20;

    /// <summary>
    /// Gets or sets the species size above which the species champion is copied unchanged into the next
    /// generation. Default: <c>5</c>.
    /// </summary>
    public int ChampionSpeciesSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the fraction of each species (best first) allowed to reproduce; the rest are
    /// eliminated. Default: <c>0.2</c> (NEAT-Python's survival_threshold; the paper eliminates "the
    /// lowest performing members" without a number).
    /// </summary>
    public double SurvivalThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the slope of the steepened sigmoid 1 / (1 + e^(-slope x)). Default: <c>4.9</c>.
    /// </summary>
    public double SigmoidSlope { get; set; } = 4.9;

    /// <summary>
    /// Gets or sets how many generations one supervised training step evolves. Default: <c>50</c>.
    /// </summary>
    public int GenerationsPerTrain { get; set; } = 50;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        RequireProbability(CrossoverRate, nameof(CrossoverRate));
        RequireProbability(WeightMutationRate, nameof(WeightMutationRate));
        RequireProbability(WeightReplaceRate, nameof(WeightReplaceRate));
        RequireProbability(AddNodeRate, nameof(AddNodeRate));
        RequireProbability(AddConnectionRate, nameof(AddConnectionRate));
        RequireProbability(DisableInheritedRate, nameof(DisableInheritedRate));
        RequireProbability(InterspeciesMatingRate, nameof(InterspeciesMatingRate));
        RequireProbability(SurvivalThreshold, nameof(SurvivalThreshold));
        if (!(CompatibilityThreshold > 0))
        {
            throw new ArgumentException(
                $"{GetType().Name}.CompatibilityThreshold is {CompatibilityThreshold}, but it must be positive.",
                OptionsParameterName);
        }
        if (!(WeightCap > 0) || WeightPerturbPower < 0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.WeightCap must be positive and WeightPerturbPower non-negative.",
                OptionsParameterName);
        }
        if (GenerationsPerTrain < 1 || AddConnectionTries < 1)
        {
            throw new ArgumentException(
                $"{GetType().Name}.GenerationsPerTrain and AddConnectionTries must be at least 1.",
                OptionsParameterName);
        }
    }

    private void RequireProbability(double value, string name)
    {
        if (value < 0.0 || value > 1.0 || double.IsNaN(value))
        {
            throw new ArgumentException(
                $"{GetType().Name}.{name} is {value}, but it must be between 0.0 and 1.0.",
                OptionsParameterName);
        }
    }
}

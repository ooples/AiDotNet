namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded with real-valued genes, suitable for numerical optimization problems.
/// </summary>
/// <remarks>
/// <para>
/// The RealValuedIndividual class implements a solution representation using a vector of real-valued genes.
/// This encoding is particularly useful for optimization problems with continuous parameters, such as
/// function optimization, parameter tuning, engineering design, or machine learning model hyperparameters.
/// </para>
/// <para><b>For Beginners:</b> Think of a RealValuedIndividual like a control panel with multiple adjustable knobs.
/// 
/// Imagine a mixing console in a recording studio:
/// - Each knob (gene) controls a different parameter (volume, bass, treble, etc.)
/// - Each knob can be set to any position within its range (continuous values)
/// - The complete setting of all knobs represents one possible solution
/// - Different knob configurations produce different results
/// - The goal is to find the configuration that produces the best sound (optimum solution)
/// 
/// During evolution, these knob settings get adjusted, combined, and fine-tuned to find
/// the best possible configuration for solving the problem at hand.
/// </para>
/// </remarks>
public class RealValuedIndividual : IEvolvable<RealGene, double>
{
    /// <summary>
    /// The collection of real-valued genes that define this individual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the vector of real-valued genes that represent the individual's solution.
    /// Each gene corresponds to a dimension in the search space, and together they define a specific
    /// point in that space.
    /// </para>
    /// <para><b>For Beginners:</b> This is the collection of all knobs on your control panel.
    /// 
    /// For example:
    /// - In a 3D optimization problem, you might have 3 genes representing x, y, and z coordinates
    /// - In a machine learning application, each gene might represent a different hyperparameter
    /// - In an engineering design, each gene might represent a different dimension or property
    /// 
    /// The combined settings of all these knobs define your complete solution,
    /// and the genetic algorithm's job is to find the optimal settings.
    /// </para>
    /// </remarks>
    private List<RealGene> _genes = [];

    /// <summary>
    /// The fitness score that indicates how well this individual solves the problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the individual's fitness score, which quantifies how well this solution
    /// performs on the given problem. Depending on the problem, higher or lower values might
    /// indicate better solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the quality rating of your control panel configuration.
    /// 
    /// For example:
    /// - In a maximization problem (like profit optimization), higher fitness is better
    /// - In a minimization problem (like error reduction), lower fitness is better
    /// - This score determines which configurations survive and reproduce
    /// - Solutions with better scores have a higher chance of passing their settings to the next generation
    /// 
    /// The fitness score is the key measure that drives the entire evolutionary process
    /// toward better solutions.
    /// </para>
    /// </remarks>
    private double _fitness;

    /// <summary>
    /// Creates a new individual with random values within the specified range.
    /// </summary>
    /// <param name="dimensionCount">The number of dimensions (genes).</param>
    /// <param name="minValue">The minimum value for initialization.</param>
    /// <param name="maxValue">The maximum value for initialization.</param>
    /// <param name="random">Random number generator for initialization.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new individual with a specified number of genes, each initialized
    /// with a random value within the given range. This is commonly used to create the initial
    /// population in a genetic algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a new control panel with all knobs at random positions.
    /// 
    /// When creating a new random individual:
    /// - You specify how many knobs (dimensions) to include
    /// - You define the allowed range for each knob (minimum to maximum)
    /// - The constructor randomly sets each knob to a position within its allowed range
    /// - Each new individual gets a different random configuration
    /// 
    /// This randomization creates diversity in the initial population, giving the algorithm
    /// a wide range of starting points to explore the solution space.
    /// </para>
    /// </remarks>
    public RealValuedIndividual(int dimensionCount, double minValue, double maxValue, Random random)
    {
        for (int i = 0; i < dimensionCount; i++)
        {
            double value = minValue + (maxValue - minValue) * random.NextDouble();
            _genes.Add(new RealGene(value));
        }
    }

    /// <summary>
    /// Creates a real-valued individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new individual with a predefined set of genes.
    /// It's typically used when creating offspring during crossover operations or
    /// when initializing individuals with known good values.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a control panel with specific, predetermined knob positions.
    /// 
    /// When creating a specific individual:
    /// - You provide the exact settings for each knob
    /// - This is often used when creating offspring during evolution
    /// - It might combine settings from two parent configurations
    /// - Or it might be used to initialize the population with known good settings
    /// 
    /// This constructor allows the algorithm to create new individuals with predetermined
    /// gene values, which is essential for genetic operations.
    /// </para>
    /// </remarks>
    public RealValuedIndividual(ICollection<RealGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the values of all genes as an array.
    /// </summary>
    /// <returns>An array of double values.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the raw values from all genes and returns them as an array of doubles.
    /// It's useful for evaluating the solution or for problem-specific operations that work
    /// directly with the numerical values rather than gene objects.
    /// </para>
    /// <para><b>For Beginners:</b> This is like reading off all the knob positions as a list of numbers.
    /// 
    /// For example:
    /// - If your control panel has 5 knobs set to positions 1.5, 3.2, -0.7, 4.0, and 2.1
    /// - This method returns the array [1.5, 3.2, -0.7, 4.0, 2.1]
    /// - This makes it easier to work with the actual numeric values
    /// - It's often used when calculating fitness or applying mathematical operations
    /// 
    /// This provides a simpler representation of the solution for operations that
    /// don't need to work with the full gene objects.
    /// </para>
    /// </remarks>
    public double[] GetValuesAsArray()
    {
        return [.. _genes.Select(g => g.Value)];
    }

    /// <summary>
    /// Updates the step sizes according to Evolutionary Strategies 1/5 success rule.
    /// </summary>
    /// <param name="successRatio">The ratio of successful mutations.</param>
    /// <remarks>
    /// <para>
    /// This method implements the 1/5 success rule from Evolutionary Strategies, a principle
    /// for adapting mutation step sizes. If more than 1/5 of mutations are successful (improve fitness),
    /// the step size increases to explore more widely. If fewer than 1/5 are successful, the step size
    /// decreases to focus the search more precisely.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adjusting how sensitive your control knobs are based on how well small adjustments are working.
    /// 
    /// Imagine you're tuning a radio:
    /// - If small adjustments frequently improve reception (high success ratio), you might want to make larger adjustments to find the optimum faster
    /// - If small adjustments rarely help (low success ratio), you're probably near an optimum and need more precise, smaller adjustments
    /// - The 1/5 rule says: aim for about 20% of your adjustments to be improvements
    /// - If more than 20% work, make bigger adjustments
    /// - If fewer than 20% work, make smaller adjustments
    /// 
    /// This adaptive approach helps balance exploration (trying varied solutions) and
    /// exploitation (refining promising solutions) throughout the evolutionary process.
    /// </para>
    /// </remarks>
    public void UpdateStepSizes(double successRatio)
    {
        const double c = 0.817; // Constant derived from theoretical considerations
        // Increase step size if success ratio is high, decrease if low
        double adjustmentFactor = successRatio > 0.2 ? 1.0 / c : c;
        foreach (var gene in _genes)
        {
            gene.StepSize *= adjustmentFactor;
        }
    }

    /// <summary>
    /// Gets the genes of this individual.
    /// </summary>
    /// <returns>The collection of genes.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the individual's genetic information as required by the IEvolvable interface.
    /// It returns the complete collection of real-valued genes that define this solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like getting access to all the knobs on your control panel.
    /// 
    /// This method:
    /// - Returns all the knobs with their current settings
    /// - Allows other parts of the algorithm to examine or manipulate the knobs
    /// - Is used during operations like crossover and mutation
    /// 
    /// This is one of the core methods required by the genetic algorithm to work
    /// with real-valued individuals.
    /// </para>
    /// </remarks>
    public ICollection<RealGene> GetGenes()
    {
        return _genes;
    }

    /// <summary>
    /// Sets the genes of this individual.
    /// </summary>
    /// <param name="genes">The collection of genes to set.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the individual's genetic information as required by the IEvolvable interface.
    /// It updates all genes with a new collection, effectively changing the entire solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like replacing all the knobs on your control panel at once.
    /// 
    /// This method:
    /// - Replaces the entire set of knobs with a new configuration
    /// - Is used when creating new offspring or applying specialized operations
    /// - Updates how this individual represents a solution to the problem
    /// 
    /// This is one of the core methods required by the genetic algorithm to modify
    /// real-valued individuals during evolution.
    /// </para>
    /// </remarks>
    public void SetGenes(ICollection<RealGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness score as a double value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the individual's fitness score as required by the IEvolvable interface.
    /// The score quantifies how well this solution performs on the given problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking the performance rating of your control panel configuration.
    /// 
    /// The fitness score:
    /// - Indicates how good this particular configuration is at solving the problem
    /// - May need to be maximized or minimized depending on the problem
    /// - Determines which configurations survive and reproduce
    /// - Drives the entire evolutionary process toward better solutions
    /// 
    /// This is one of the core methods required by the genetic algorithm to evaluate
    /// and compare different solutions.
    /// </para>
    /// </remarks>
    public double GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The fitness score to set.</param>
    /// <remarks>
    /// <para>
    /// This method assigns the fitness score to the individual as required by the IEvolvable interface.
    /// It is typically called after evaluating the quality of the solution for the specific problem.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recording the performance rating for your control panel configuration.
    /// 
    /// After testing how well your configuration performs:
    /// - The score is calculated based on problem-specific criteria
    /// - This method stores that score with the configuration
    /// - Later, this score will be used to compare different configurations
    /// - Better-scoring configurations have a higher chance of influencing the next generation
    /// 
    /// This is one of the core methods required by the genetic algorithm to track
    /// the quality of different solutions during evolution.
    /// </para>
    /// </remarks>
    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates a deep copy of this individual.
    /// </summary>
    /// <returns>A new RealValuedIndividual that is a copy of this one.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the individual, including all its genes
    /// and its fitness score. It's essential for genetic operations where individuals need to be
    /// copied without modifying the originals.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an exact duplicate of your entire control panel.
    /// 
    /// When cloning an individual:
    /// - A completely new control panel is created
    /// - Each knob is copied with the same position and sensitivity settings
    /// - The performance rating is also copied
    /// - Changes to one panel won't affect the other
    /// 
    /// This operation is crucial in genetic algorithms for preserving good solutions
    /// while experimenting with modifications to create potentially better ones.
    /// </para>
    /// </remarks>
    public IEvolvable<RealGene, double> Clone()
    {
        var clone = new RealValuedIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;

        return clone;
    }
}
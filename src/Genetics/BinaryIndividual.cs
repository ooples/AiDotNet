namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual encoded with binary genes, suitable for classic GA problems.
/// </summary>
/// <remarks>
/// <para>
/// BinaryIndividual implements the IEvolvable interface to participate in genetic evolution processes.
/// It represents a complete chromosome composed of binary genes, where each gene is either 0 or 1.
/// This implementation is particularly useful for classic genetic algorithm problems where solutions
/// can be effectively represented as binary strings.
/// </para>
/// <para><b>For Beginners:</b> Think of a BinaryIndividual as a row of light switches.
/// 
/// In genetic algorithms:
/// - A BinaryIndividual is like a complete row of on/off switches (where each switch is a BinaryGene)
/// - The pattern of on/off positions represents a potential solution to a problem
/// - The "fitness" score tells us how good this particular pattern is at solving our problem
/// - During evolution, these patterns compete, combine, and mutate to find better solutions
/// 
/// Binary encoding is one of the simplest and most common ways to represent solutions in genetic algorithms,
/// making it a good starting point for understanding evolutionary computation.
/// </para>
/// </remarks>
public class BinaryIndividual : IEvolvable<BinaryGene, double>
{
    /// <summary>
    /// The collection of binary genes that form this individual's chromosome.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the actual genetic information of the individual as a list of BinaryGene objects.
    /// Each gene represents a bit in the binary string that encodes the solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual row of switches that makes up this individual.
    /// Each switch (gene) can be either on or off, and the entire pattern of switches together
    /// represents a potential solution to the problem being solved.
    /// </para>
    /// </remarks>
    private List<BinaryGene> _genes = [];

    /// <summary>
    /// The fitness score that indicates how well this individual solves the problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the individual's fitness score, which quantifies how good this solution is.
    /// Higher fitness values typically indicate better solutions, depending on the fitness function used.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a score that rates how good this particular pattern
    /// of switches is at solving the problem. The higher the fitness, the better this individual
    /// is as a solution, and the more likely it will be selected to pass its genes to the next generation.
    /// </para>
    /// </remarks>
    private double _fitness;

    /// <summary>
    /// Creates a new binary individual with the specified chromosome length.
    /// </summary>
    /// <param name="length">The number of genes (bits) in the chromosome.</param>
    /// <param name="random">Random number generator for initialization.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BinaryIndividual with a randomized chromosome of the specified length.
    /// Each gene in the chromosome is randomly initialized to either 0 or 1 using the provided random number generator.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a new row of switches with a random pattern.
    /// 
    /// When creating a new individual this way:
    /// - You specify how many switches you want in the row (the length)
    /// - You provide a random number generator to decide the position of each switch
    /// - Each switch is randomly set to either on or off
    /// - The result is a completely random potential solution
    /// 
    /// Random initialization is important for creating diverse starting populations
    /// in genetic algorithms.
    /// </para>
    /// </remarks>
    public BinaryIndividual(int length, Random random)
    {
        for (int i = 0; i < length; i++)
        {
            _genes.Add(new BinaryGene(random.Next(2))); // Randomly 0 or 1
        }
    }

    /// <summary>
    /// Creates a binary individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new BinaryIndividual with a chromosome composed of the provided genes.
    /// It makes a copy of the input collection to ensure the individual has its own independent genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a row of switches with a specific pattern.
    /// 
    /// When creating a new individual this way:
    /// - You provide the exact pattern of on/off switches you want
    /// - The individual copies this pattern to create its own row of switches
    /// - This allows you to create individuals with specific characteristics
    /// 
    /// This constructor is often used when creating offspring during crossover operations
    /// or when you want to initialize a population with known good solutions.
    /// </para>
    /// </remarks>
    public BinaryIndividual(ICollection<BinaryGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the binary value of this individual as an integer.
    /// </summary>
    /// <returns>The integer value represented by this binary string.</returns>
    /// <remarks>
    /// <para>
    /// This method interprets the binary chromosome as an unsigned integer and returns its value.
    /// Each gene in the chromosome is treated as a bit in the integer, with the position in the
    /// chromosome determining the bit position (least significant bit first).
    /// </para>
    /// <para><b>For Beginners:</b> This converts the pattern of switches into a single number.
    /// 
    /// For example, if you have 8 switches:
    /// - Each switch position represents a power of 2 (1, 2, 4, 8, 16, 32, 64, 128)
    /// - If a switch is on (1), its value is added to the total
    /// - If a switch is off (0), nothing is added
    /// - So the pattern [1,0,1,1,0,0,0,0] would equal 1+0+4+8+0+0+0+0 = 13
    /// 
    /// This method is useful when you need to convert the binary representation
    /// into a numeric value for fitness calculation or problem solving.
    /// </para>
    /// </remarks>
    public int GetValueAsInt()
    {
        int value = 0;
        for (int i = 0; i < _genes.Count; i++)
        {
            if (_genes[i].Value == 1)
            {
                value |= (1 << i);
            }
        }
        return value;
    }

    /// <summary>
    /// Gets the binary value of this individual as a double in the range [0,1].
    /// </summary>
    /// <returns>A double value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the binary chromosome to a normalized double value between 0 and 1.
    /// It first converts the binary string to an integer, then divides by the maximum possible value
    /// that could be represented with the current chromosome length.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the pattern of switches into a percentage or decimal.
    /// 
    /// For example:
    /// - If all switches are off, the result is 0.0 (0%)
    /// - If all switches are on, the result is 1.0 (100%)
    /// - Any other pattern gives a value somewhere in between
    /// - The more bits (switches) you have, the more precisely you can represent values
    /// 
    /// This normalization is useful when you need to map the binary representation
    /// to continuous domains or when comparing individuals with different chromosome lengths.
    /// </para>
    /// </remarks>
    public double GetValueAsNormalizedDouble()
    {
        double maxValue = Math.Pow(2, _genes.Count) - 1;
        return GetValueAsInt() / maxValue;
    }

    /// <summary>
    /// Maps the binary string to a double value within the specified range.
    /// </summary>
    /// <param name="min">The minimum value of the range.</param>
    /// <param name="max">The maximum value of the range.</param>
    /// <returns>A double value between min and max.</returns>
    /// <remarks>
    /// <para>
    /// This method maps the binary chromosome to a double value within the specified range [min, max].
    /// It first normalizes the binary value to [0,1], then scales and shifts it to the target range.
    /// This is particularly useful when the binary chromosome represents a real-valued parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is like using the switches to represent a value within a specific range.
    /// 
    /// For example, if you want to represent a value between 10 and 20:
    /// - The pattern with all switches off would represent exactly 10
    /// - The pattern with all switches on would represent exactly 20
    /// - Other patterns would give values in between, spread evenly across the range
    /// 
    /// This mapping is extremely useful in genetic algorithms for parameter optimization,
    /// where you need to search through continuous ranges of values.
    /// </para>
    /// </remarks>
    public double GetValueMapped(double min, double max)
    {
        return min + (GetValueAsNormalizedDouble() * (max - min));
    }

    /// <summary>
    /// Gets the collection of genes that make up this individual's chromosome.
    /// </summary>
    /// <returns>The collection of binary genes.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the individual's genetic information.
    /// It returns the complete collection of binary genes that constitute the chromosome.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you access to the entire row of switches.
    /// 
    /// You might use this method when you need to:
    /// - Examine the specific pattern of on/off values
    /// - Apply genetic operations like mutation or crossover
    /// - Analyze or visualize the chromosome structure
    /// 
    /// This is one of the core interface methods that enables the genetic algorithm
    /// to work with this individual during the evolutionary process.
    /// </para>
    /// </remarks>
    public ICollection<BinaryGene> GetGenes()
    {
        return _genes;
    }

    /// <summary>
    /// Sets the genes for this individual's chromosome.
    /// </summary>
    /// <param name="genes">The collection of genes to set.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the individual's genetic information with the provided collection of genes.
    /// It creates a copy of the input collection to ensure the individual has its own independent genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like replacing the entire row of switches with a new pattern.
    /// 
    /// You might use this method when:
    /// - Creating a new offspring after crossover
    /// - Applying special genetic operations that modify the entire chromosome
    /// - Resetting an individual to a known state
    /// 
    /// This is one of the core interface methods that enables the genetic algorithm
    /// to modify this individual during the evolutionary process.
    /// </para>
    /// </remarks>
    public void SetGenes(ICollection<BinaryGene> genes)
    {
        _genes = [.. genes];
    }

    /// <summary>
    /// Gets the fitness score of this individual.
    /// </summary>
    /// <returns>The fitness score as a double value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the individual's fitness score, which quantifies how well this solution performs.
    /// The fitness score is typically set by a fitness function that evaluates the individual against the problem criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how good this pattern of switches is at solving the problem.
    /// 
    /// The fitness score:
    /// - Is calculated by testing how well this specific pattern performs
    /// - Allows different individuals to be compared to each other
    /// - Determines which individuals are more likely to reproduce
    /// - Guides the entire evolutionary process toward better solutions
    /// 
    /// This is one of the most important attributes in genetic algorithms, as it
    /// drives the selection process and measures progress toward a solution.
    /// </para>
    /// </remarks>
    public double GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness score for this individual.
    /// </summary>
    /// <param name="fitness">The fitness score to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets the individual's fitness score after evaluation.
    /// It is typically called by the genetic algorithm after applying a fitness function to the individual.
    /// </para>
    /// <para><b>For Beginners:</b> This is like assigning a grade to how well this pattern of switches performs.
    /// 
    /// After testing this individual:
    /// - The genetic algorithm calculates a score based on how well it solves the problem
    /// - That score is stored here so that the individual can be compared with others
    /// - Higher scores typically indicate better solutions
    /// 
    /// Setting and accessing fitness scores is crucial for selection mechanisms
    /// that determine which individuals get to reproduce.
    /// </para>
    /// </remarks>
    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates an independent copy of this individual.
    /// </summary>
    /// <returns>A new BinaryIndividual with the same genes and fitness score.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the individual, including all its genes and its fitness score.
    /// Each gene is individually cloned to ensure the new individual has completely independent genetic information.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an exact duplicate of this individual.
    /// 
    /// When you clone an individual:
    /// - You get a completely separate copy with its own row of switches
    /// - Each switch is individually duplicated, maintaining the exact same pattern
    /// - The fitness score is also copied
    /// - Changes to one individual won't affect the other
    /// 
    /// Cloning is essential in genetic algorithms for operations like elitism
    /// (preserving the best individuals) and for creating starting points for
    /// mutation and crossover operations.
    /// </para>
    /// </remarks>
    public IEvolvable<BinaryGene, double> Clone()
    {
        var clone = new BinaryIndividual([]);
        foreach (var gene in _genes)
        {
            clone._genes.Add(gene.Clone());
        }
        clone._fitness = _fitness;
        return clone;
    }
}
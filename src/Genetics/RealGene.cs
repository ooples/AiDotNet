namespace AiDotNet.Genetics;

/// <summary>
/// Represents a gene with a real (double) value.
/// </summary>
/// <remarks>
/// <para>
/// The RealGene class implements a gene that holds a continuous, real-valued number (double).
/// This type of gene is commonly used in evolutionary algorithms for problems where solutions
/// are represented by continuous parameters, such as function optimization, neural network
/// weight optimization, or engineering design problems.
/// </para>
/// <para><b>For Beginners:</b> Think of a RealGene like a volume knob on a stereo.
/// 
/// While some genes are like on/off switches (binary genes) or like selecting specific cards 
/// (permutation genes), real genes are more like knobs or sliders that can be turned to any position:
/// - The Value is the current position of the knob (e.g., 42.5)
/// - The knob can be turned to any position within its range, not just fixed stops
/// - The StepSize is like how much the knob typically moves when adjusted (small or large increments)
/// 
/// Real-valued genes are perfect for problems where you need to find precise numerical 
/// values, like finding the optimal dimensions for a bridge design or the best settings 
/// for a controller.
/// </para>
/// </remarks>
public class RealGene
{
    /// <summary>
    /// Gets or sets the real value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the actual double-precision floating-point value of the gene.
    /// It represents a point in the continuous search space and is the primary value that is
    /// manipulated during evolutionary operations like mutation and crossover.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual position of the volume knob.
    /// 
    /// For example:
    /// - A value of 3.14159 might represent a specific dimension in a design problem
    /// - A value of -0.5 might represent a weight in a neural network
    /// - A value of 42.0 might represent a temperature in a process optimization problem
    /// 
    /// This value directly affects the behavior or performance of the solution being evolved,
    /// and finding optimal values for these genes is the main goal of the evolutionary algorithm.
    /// </para>
    /// </remarks>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the mutation step size (used in evolutionary strategies).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the standard deviation or typical change size used when mutating this gene.
    /// It's an essential part of self-adaptive evolutionary strategies, where the mutation step size
    /// itself evolves along with the solution, allowing the algorithm to automatically adjust
    /// the exploration vs. exploitation balance.
    /// </para>
    /// <para><b>For Beginners:</b> This is like how sensitive the volume knob is when you turn it.
    /// 
    /// For example:
    /// - A large step size (e.g., 1.0) means big jumps when changing the value (coarse adjustment)
    /// - A small step size (e.g., 0.01) means tiny changes (fine tuning)
    /// - As evolution progresses, step sizes often get smaller as the algorithm refines its search
    /// - Some advanced algorithms let this sensitivity evolve too, automatically finding the right balance
    /// 
    /// This adaptive behavior helps the algorithm explore widely early on and then focus more
    /// precisely as it gets closer to good solutions.
    /// </para>
    /// </remarks>
    public double StepSize { get; set; }

    /// <summary>
    /// Initializes a new instance of the RealGene class with optional initial value and step size.
    /// </summary>
    /// <param name="value">The initial value for the gene (default: 0.0).</param>
    /// <param name="stepSize">The initial mutation step size (default: 0.1).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RealGene with the specified value and step size.
    /// Default values are provided, making it easy to create genes with standard settings.
    /// The step size default of 0.1 is a common starting point that balances exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a new volume knob with an initial position and sensitivity.
    /// 
    /// When creating a new gene:
    /// - You can specify where the knob starts (the initial value)
    /// - You can specify how sensitive it is to adjustments (the step size)
    /// - If you don't specify, it starts at 0.0 with a medium sensitivity (0.1)
    /// 
    /// This setup creates a new gene that's ready to be used in an evolutionary algorithm,
    /// with reasonable defaults if you're not sure what values to use.
    /// </para>
    /// </remarks>
    public RealGene(double value = 0.0, double stepSize = 0.1)
    {
        Value = value;
        StepSize = stepSize;
    }

    /// <summary>
    /// Creates an independent copy of this gene.
    /// </summary>
    /// <returns>A new RealGene with the same value and step size.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new RealGene that is an exact copy of the current one.
    /// Cloning is essential in genetic algorithms for creating offspring without modifying
    /// the original genes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like making an exact duplicate of a volume knob.
    /// 
    /// When cloning a gene:
    /// - You get a completely new gene object
    /// - It has the same value (knob position) as the original
    /// - It has the same step size (sensitivity) as the original
    /// - Changes to the clone won't affect the original
    /// 
    /// This operation is used during genetic operations like crossover,
    /// allowing genes to be copied between individuals without changing the originals.
    /// </para>
    /// </remarks>
    public RealGene Clone()
    {
        return new RealGene(Value, StepSize);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current gene.
    /// </summary>
    /// <param name="obj">The object to compare with the current gene.</param>
    /// <returns>true if the specified object is a RealGene with the same value and step size; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base Equals method to provide value-based equality comparison.
    /// Two RealGene objects are considered equal if their Value and StepSize properties are
    /// very close (within a small epsilon to account for floating-point precision issues).
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking if two volume knobs are set to the same position with the same sensitivity.
    /// 
    /// When comparing two genes:
    /// - It checks if the other object is also a RealGene
    /// - It checks if both values are essentially the same (accounting for tiny floating-point differences)
    /// - It checks if both step sizes are essentially the same
    /// - Only if all conditions are true, the genes are considered equal
    /// 
    /// This method helps detect duplicate genes or compare genes before and after operations.
    /// </para>
    /// </remarks>
    public override bool Equals(object? obj)
    {
        return obj is RealGene gene &&
               Math.Abs(gene.Value - Value) < 1e-10 &&
               Math.Abs(gene.StepSize - StepSize) < 1e-10;
    }

    /// <summary>
    /// Returns a hash code for this gene.
    /// </summary>
    /// <returns>A hash code for the current gene.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base GetHashCode method to ensure that the hash code
    /// is consistent with the Equals method. It combines the hash codes of both the Value
    /// and StepSize properties to create a unique hash code for the gene.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique fingerprint for the volume knob.
    /// 
    /// The hash code:
    /// - Provides a number that can be used to quickly compare or identify genes
    /// - Is calculated based on both the position and sensitivity of the knob
    /// - Ensures that identical genes get the same hash code
    /// - Helps when storing genes in collections like dictionaries or hash sets
    /// 
    /// While not directly used in genetic operations, this method is important for
    /// the proper functioning of many collections and algorithms.
    /// </para>
    /// </remarks>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + Value.GetHashCode();
            hash = hash * 23 + StepSize.GetHashCode();

            return hash;
        }
    }
}
namespace AiDotNet.Genetics;

/// <summary>
/// A real-valued individual supporting multi-objective optimization.
/// </summary>
/// <remarks>
/// <para>
/// The MultiObjectiveRealIndividual class extends the standard RealValuedIndividual to support
/// optimization problems with multiple competing objectives. It implements the IMultiObjectiveIndividual
/// interface and introduces concepts such as objective values, dominance ranking, and crowding distance
/// which are essential for multi-objective evolutionary algorithms like NSGA-II.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a versatile candidate solution that can be evaluated on multiple criteria.
/// 
/// In regular optimization, we're looking for the best solution to a single problem:
/// - Like finding the fastest route from A to B
/// 
/// But in multi-objective optimization, we're balancing multiple goals that often conflict:
/// - Like finding a route that's fast, uses minimal fuel, AND avoids toll roads
/// - There's usually no single "best" solution that wins in all categories
/// - Instead, we find a set of solutions where each one represents a different tradeoff
/// 
/// This class enables genetic algorithms to handle these complex, multi-goal problems by tracking how 
/// good a solution is across each objective and organizing solutions into tiers (ranks) based on which
/// solutions are better than others across all objectives.
/// </para>
/// </remarks>
public class MultiObjectiveRealIndividual : RealValuedIndividual, IMultiObjectiveIndividual<double>
{
    /// <summary>
    /// The values achieved by this individual for each objective function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the performance of the individual across multiple objective functions.
    /// Each value represents how well the individual performs on a specific objective, with lower
    /// values typically representing better performance (minimization).
    /// </para>
    /// <para><b>For Beginners:</b> These are the individual's scores on each separate goal.
    /// 
    /// For example, if optimizing a vehicle design:
    /// - First value might be fuel consumption (lower is better)
    /// - Second value might be manufacturing cost (lower is better)
    /// - Third value might be safety rating (higher is better, but often inverted in calculations)
    /// 
    /// These values are used to determine if one solution is better than another
    /// and to organize solutions into tiers of quality.
    /// </para>
    /// </remarks>
    private List<double> _objectiveValues = [];

    /// <summary>
    /// The dominance rank of this individual in the population.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the rank of the individual based on Pareto dominance relationships.
    /// Individuals with rank 0 form the first Pareto front (non-dominated solutions),
    /// rank 1 forms the second front, and so on. Lower ranks indicate better solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the tier or league the solution belongs to.
    /// 
    /// Think of it as gold, silver, bronze medals, etc.:
    /// - Rank 0 solutions are the "gold medal tier" - no solution is better in all objectives
    /// - Rank 1 solutions are the "silver medal tier" - only dominated by gold solutions
    /// - And so on...
    /// 
    /// During selection, solutions with lower ranks are preferred, as they represent
    /// better trade-offs among the multiple objectives.
    /// </para>
    /// </remarks>
    private int _rank;

    /// <summary>
    /// The crowding distance of this individual in its Pareto front.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a measure of how close this individual is to its neighbors in the same Pareto front.
    /// Higher values indicate more isolated individuals, which are often prioritized in selection to
    /// maintain diversity along the Pareto front.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how unique a solution is compared to similar ones.
    /// 
    /// Imagine solutions arranged on a chart:
    /// - Solutions that are bunched together offer similar trade-offs
    /// - Solutions that stand apart offer unique trade-offs
    /// - We want to keep unique solutions to give users diverse options
    /// 
    /// The crowding distance helps the algorithm maintain diversity by favoring
    /// solutions that represent unique trade-offs within their rank.
    /// </para>
    /// </remarks>
    private double _crowdingDistance;

    /// <summary>
    /// Creates a new multi-objective individual with random gene values within the specified range.
    /// </summary>
    /// <param name="dimensionCount">The number of genes (dimensions) for this individual.</param>
    /// <param name="minValue">The minimum value for each gene.</param>
    /// <param name="maxValue">The maximum value for each gene.</param>
    /// <param name="random">The random number generator to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new multi-objective individual with random gene values.
    /// It delegates to the base class constructor to create the random genes within the specified range.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new solution with random values for each characteristic.
    /// 
    /// For example, when creating a random vehicle design:
    /// - You specify how many design parameters to include (dimensionCount)
    /// - You set the minimum and maximum values for each parameter
    /// - The constructor randomly assigns a value to each parameter within these bounds
    /// 
    /// This creates a starting point for the evolutionary algorithm to begin exploring
    /// the solution space.
    /// </para>
    /// </remarks>
    public MultiObjectiveRealIndividual(int dimensionCount, double minValue, double maxValue, Random random)
        : base(dimensionCount, minValue, maxValue, random)
    {
    }

    /// <summary>
    /// Creates a new multi-objective individual with the specified genes.
    /// </summary>
    /// <param name="genes">The collection of genes to initialize with.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new multi-objective individual with the provided genes.
    /// It delegates to the base class constructor to set up the individual with these genes.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new solution with specific values for each characteristic.
    /// 
    /// For example, when creating a specific vehicle design:
    /// - You provide the exact values for each design parameter
    /// - This is often used when creating offspring during crossover operations
    /// - Or when copying an existing good solution
    /// 
    /// This constructor allows the algorithm to create new individuals with predetermined
    /// genetic information, which is essential for genetic operations.
    /// </para>
    /// </remarks>
    public MultiObjectiveRealIndividual(ICollection<RealGene> genes)
        : base(genes)
    {
    }

    /// <summary>
    /// Gets the values achieved by this individual for each objective function.
    /// </summary>
    /// <returns>A collection of objective values.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the performance values of the individual across all objectives.
    /// These values are used to compare individuals and determine dominance relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This returns the solution's scores on all the goals.
    /// 
    /// For example, for a vehicle design, it might return:
    /// - Fuel consumption: 5.2 liters/100km
    /// - Manufacturing cost: $15,000
    /// - Safety rating: 4.8/5
    /// 
    /// These scores allow the algorithm to compare different solutions and determine
    /// which ones represent better trade-offs.
    /// </para>
    /// </remarks>
    public ICollection<double> GetObjectiveValues()
    {
        return _objectiveValues;
    }

    /// <summary>
    /// Sets the objective values for this individual.
    /// </summary>
    /// <param name="values">The collection of objective values to set.</param>
    /// <remarks>
    /// <para>
    /// This method assigns the performance values across all objectives for this individual.
    /// These values are calculated by evaluating the individual against each objective function.
    /// </para>
    /// <para><b>For Beginners:</b> This sets the solution's scores on all the goals.
    /// 
    /// After evaluating how well a solution performs:
    /// - The algorithm calculates a score for each objective
    /// - These scores are stored in the individual using this method
    /// - Later, these scores determine if this solution is better than others
    /// 
    /// Setting accurate objective values is crucial for the proper functioning of
    /// multi-objective optimization algorithms.
    /// </para>
    /// </remarks>
    public void SetObjectiveValues(ICollection<double> values)
    {
        _objectiveValues = [.. values]; // Fixed from original code which incorrectly created an empty list
    }

    /// <summary>
    /// Gets the dominance rank of this individual.
    /// </summary>
    /// <returns>The rank of the individual (0 = first Pareto front).</returns>
    /// <remarks>
    /// <para>
    /// This method returns the Pareto dominance rank of the individual in the population.
    /// The rank indicates which Pareto front the individual belongs to (0 for the first front,
    /// 1 for the second front, etc.).
    /// </para>
    /// <para><b>For Beginners:</b> This returns the tier or league the solution belongs to.
    /// 
    /// A lower rank number means a better solution:
    /// - Rank 0: "Gold tier" solutions - not dominated by any other solution
    /// - Rank 1: "Silver tier" solutions - only dominated by gold tier solutions
    /// - And so on...
    /// 
    /// This ranking system allows the algorithm to prioritize solutions that represent
    /// better trade-offs among the competing objectives.
    /// </para>
    /// </remarks>
    public int GetRank()
    {
        return _rank;
    }

    /// <summary>
    /// Sets the dominance rank of this individual.
    /// </summary>
    /// <param name="rank">The rank to assign (0 = first Pareto front).</param>
    /// <remarks>
    /// <para>
    /// This method assigns the Pareto dominance rank to the individual. The rank is typically
    /// calculated by a non-dominated sorting algorithm that analyzes the entire population.
    /// </para>
    /// <para><b>For Beginners:</b> This assigns the tier or league the solution belongs to.
    /// 
    /// During the ranking process:
    /// - The algorithm compares all solutions against each other
    /// - It identifies which solutions are not dominated by any others (rank 0)
    /// - Then identifies solutions only dominated by rank 0 solutions (rank 1)
    /// - And continues until all solutions are ranked
    /// 
    /// This ranking is crucial for selection in multi-objective algorithms, as
    /// it determines which solutions are prioritized for reproduction.
    /// </para>
    /// </remarks>
    public void SetRank(int rank)
    {
        _rank = rank;
    }

    /// <summary>
    /// Gets the crowding distance of this individual in its Pareto front.
    /// </summary>
    /// <returns>The crowding distance value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the crowding distance of the individual, which measures how close it is
    /// to other individuals in the same Pareto front. Higher values indicate more isolated individuals.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how unique this solution is compared to similar ones.
    /// 
    /// The crowding distance:
    /// - Is higher for solutions that offer unique trade-offs
    /// - Is lower for solutions that are similar to many others
    /// - Helps maintain diversity by favoring unique solutions
    /// 
    /// This value is particularly important when choosing between solutions of the same rank,
    /// as it helps preserve a diverse set of trade-offs for the decision-maker.
    /// </para>
    /// </remarks>
    public double GetCrowdingDistance()
    {
        return _crowdingDistance;
    }

    /// <summary>
    /// Sets the crowding distance of this individual.
    /// </summary>
    /// <param name="distance">The crowding distance value to set.</param>
    /// <remarks>
    /// <para>
    /// This method assigns the crowding distance to the individual. The distance is typically
    /// calculated by a crowding distance assignment algorithm that analyzes all individuals
    /// in the same Pareto front.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how unique this solution is compared to similar ones.
    /// 
    /// During the diversity assessment:
    /// - The algorithm groups solutions by their rank
    /// - Within each rank, it measures how different each solution is from its neighbors
    /// - Solutions offering unique trade-offs get higher crowding distances
    /// 
    /// This value helps the algorithm maintain a diverse set of solutions along each
    /// Pareto front, providing the decision-maker with a variety of trade-off options.
    /// </para>
    /// </remarks>
    public void SetCrowdingDistance(double distance)
    {
        _crowdingDistance = distance;
    }

    /// <summary>
    /// Checks if this individual dominates another individual.
    /// </summary>
    /// <param name="other">The other individual to compare with.</param>
    /// <returns>True if this individual dominates the other, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method determines if this individual Pareto-dominates another individual.
    /// An individual dominates another if it is at least as good in all objectives and
    /// strictly better in at least one objective. For minimization problems, lower values
    /// are considered better.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if this solution is definitely better than another one.
    /// 
    /// A solution dominates another if:
    /// - It's at least as good in every single goal
    /// - AND it's definitely better in at least one goal
    /// 
    /// For example, if solution A uses less fuel, costs the same, and has better safety than
    /// solution B, then A dominates B. But if A uses less fuel but costs more, then neither
    /// dominates the other - they represent different trade-offs.
    /// 
    /// This concept is fundamental to multi-objective optimization, as it helps identify
    /// which solutions offer the best possible trade-offs (the Pareto optimal set).
    /// </para>
    /// </remarks>
    public bool Dominates(MultiObjectiveRealIndividual other)
    {
        bool atLeastOneBetter = false;
        for (int i = 0; i < _objectiveValues.Count; i++)
        {
            if (_objectiveValues[i] > other._objectiveValues[i])
            {
                return false; // This individual is worse in at least one objective
            }
            if (_objectiveValues[i] < other._objectiveValues[i])
            {
                atLeastOneBetter = true; // This individual is better in at least one objective
            }
        }
        return atLeastOneBetter;
    }
}
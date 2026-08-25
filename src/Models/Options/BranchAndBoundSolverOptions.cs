namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the branch-and-bound integer-programming solver.
/// </summary>
/// <remarks>
/// <para><b>Reference:</b> Land and Doig, “An Automatic Method of Solving Discrete Programming
/// Problems” (1960).</para>
/// <para><b>For Beginners:</b> The solver divides a whole-number problem into smaller branches,
/// discarding any branch that cannot beat the best answer already found.</para>
/// </remarks>
public class BranchAndBoundSolverOptions : ModelOptions
{
    public BranchAndBoundSolverOptions() { }

    public BranchAndBoundSolverOptions(BranchAndBoundSolverOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        MaxNodes = other.MaxNodes;
        IntegralityTolerance = other.IntegralityTolerance;
        MinimumImprovement = other.MinimumImprovement;
        RelaxationOptions = new SimplexSolverOptions(other.RelaxationOptions);
    }

    /// <summary>
    /// Gets or sets the maximum number of search-tree nodes explored.
    /// </summary>
    /// <value>The node limit, defaulting to 100000.</value>
    /// <remarks>
    /// <para>
    /// Integer programming is NP-hard, so the tree can in principle be exponential in the number of
    /// integer variables. Exhausting this budget returns the best whole-number solution found so
    /// far, reported as <see cref="Solvers.LinearProgramming.LinearProgramStatus.IterationLimit"/>
    /// rather than as a certified optimum.
    /// </para>
    /// <para><b>For Beginners:</b> How many candidate branches the search may explore before giving
    /// up. Raising it can find better answers on hard problems, at the cost of time.
    /// </para>
    /// </remarks>
    public int MaxNodes { get; set; } = 100000;

    /// <summary>
    /// Gets or sets how far from a whole number a value may sit and still count as integral.
    /// </summary>
    /// <value>The integrality tolerance, defaulting to 1e-7.</value>
    /// <remarks>
    /// <para>
    /// The relaxation is solved in floating point, so a variable that is genuinely 3 can arrive as
    /// 2.9999999997. Without a tolerance the solver would branch on it forever, splitting a range
    /// that contains no integer other than the one it already has.
    /// </para>
    /// <para><b>For Beginners:</b> How close to a whole number is close enough to be treated as one.
    /// </para>
    /// </remarks>
    public double IntegralityTolerance { get; set; } = 1e-7;

    /// <summary>
    /// Gets or sets the improvement a candidate must show over the incumbent before it replaces it.
    /// </summary>
    /// <value>The improvement threshold, defaulting to 1e-9.</value>
    /// <remarks>
    /// <para>
    /// Also used for pruning: a node whose relaxation bound is not better than the incumbent by at
    /// least this much cannot contain a better integer solution, so its whole subtree is discarded.
    /// This bounding step is what makes branch and bound practical rather than exhaustive.
    /// </para>
    /// <para><b>For Beginners:</b> If a whole branch of possibilities cannot possibly beat the best
    /// answer already found, the solver throws the entire branch away without exploring it. This
    /// setting is how much better a branch must promise to be before it is worth looking at.
    /// </para>
    /// </remarks>
    public double MinimumImprovement { get; set; } = 1e-9;

    /// <summary>
    /// Gets or sets the options used for each linear-programming relaxation solved at a node.
    /// </summary>
    /// <value>A separately owned copy of the inner simplex-solver settings.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every node of the search solves an ordinary linear program.
    /// These are the settings for those inner solves.
    /// </para>
    /// </remarks>
    public SimplexSolverOptions RelaxationOptions { get; set; } = new();
}

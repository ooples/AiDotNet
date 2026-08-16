using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers.LinearProgramming;

/// <summary>
/// A linear program in which some or all variables are required to take whole-number values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adding integrality to a linear program changes its character completely. A linear program is
/// solvable in polynomial time; deciding integer feasibility is NP-complete (Karp, 1972), so every
/// exact method is a search whose worst case is exponential. What makes it tractable in practice is
/// that the linear relaxation — the same problem with the integrality dropped — is cheap to solve
/// and gives a bound strong enough to prune most of the search tree.
/// </para>
/// <para><b>For Beginners:</b> Some quantities cannot be fractional. You can buy 2.7 kilos of
/// flour, but you cannot hire 2.7 people, run 2.7 delivery vans, or open 2.7 warehouses. Rounding
/// the fractional answer is tempting and frequently wrong — the rounded point is often infeasible,
/// and when it is feasible it can be far from the best whole-number choice. Stating the
/// requirement up front lets the solver search whole-number answers properly.
/// </para>
/// <example>
/// A knapsack: pick items to maximize value without exceeding a weight limit, taking each item
/// zero or one times.
/// <code>
/// var relaxation = new LinearProgram&lt;double&gt;(
///     objective: values.Negated(),        // maximize value
///     inequalityMatrix: weightsAsOneRow,
///     inequalityBounds: capacity,
///     lowerBounds: zeros,
///     upperBounds: ones);
///
/// var program = new IntegerProgram&lt;double&gt;(relaxation);   // every variable integral
/// </code>
/// </example>
/// </remarks>
public sealed class IntegerProgram<T>
{
    /// <summary>Gets the underlying linear program, ignoring the integrality requirements.</summary>
    public LinearProgram<T> Relaxation { get; }

    /// <summary>
    /// Gets a flag per variable indicating whether that variable must take an integer value.
    /// </summary>
    public IReadOnlyList<bool> IntegralityMask { get; }

    /// <summary>Gets the number of decision variables.</summary>
    public int VariableCount => Relaxation.VariableCount;

    /// <summary>
    /// Creates an integer program.
    /// </summary>
    /// <param name="relaxation">The underlying linear program.</param>
    /// <param name="integralityMask">
    /// One flag per variable: <c>true</c> where the variable must be integral. When omitted, every
    /// variable is required to be integral (a pure integer program).
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="relaxation"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="integralityMask"/> has a different length from the number of
    /// variables, or when it marks no variable as integral (in which case the problem is a plain
    /// linear program and should be solved as one).
    /// </exception>
    public IntegerProgram(LinearProgram<T> relaxation, IReadOnlyList<bool>? integralityMask = null)
    {
        if (relaxation is null) throw new ArgumentNullException(nameof(relaxation));

        if (integralityMask is null)
        {
            var allIntegral = new bool[relaxation.VariableCount];
            for (int i = 0; i < allIntegral.Length; i++) allIntegral[i] = true;
            integralityMask = allIntegral;
        }
        else
        {
            if (integralityMask.Count != relaxation.VariableCount)
            {
                throw new ArgumentException(
                    $"The integrality mask has {integralityMask.Count} entries but the program has " +
                    $"{relaxation.VariableCount} variables.", nameof(integralityMask));
            }

            bool anyIntegral = false;
            for (int i = 0; i < integralityMask.Count && !anyIntegral; i++)
            {
                anyIntegral = integralityMask[i];
            }

            if (!anyIntegral)
            {
                throw new ArgumentException(
                    "No variable is marked integral. Solve this as a linear program with " +
                    "ILinearProgramSolver instead — branch and bound would add cost for nothing.",
                    nameof(integralityMask));
            }
        }

        Relaxation = relaxation;
        var ownedMask = new bool[integralityMask.Count];
        for (int i = 0; i < integralityMask.Count; i++) ownedMask[i] = integralityMask[i];
        IntegralityMask = ownedMask;
    }
}

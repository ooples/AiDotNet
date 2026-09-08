using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.PDEs;

/// <summary>
/// Pins the solution to a fixed value on the boundary: <c>u = value</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Five PINN types take <c>IBoundaryCondition&lt;T&gt;[]</c> in their constructors and the library
/// implemented the interface zero times, so every user had to write this before they could solve
/// anything (#2099). A physics-informed network without boundary conditions is not solving a
/// boundary-value problem, so this is the common case rather than an edge one.
/// </para>
/// <para>
/// <b>For Beginners:</b> A differential equation on its own has infinitely many solutions — it says how
/// things change, not where they start or end. A Dirichlet condition supplies one of the missing facts:
/// "at the edge of the domain, the value is this". Holding a metal rod's end at 0°C is a Dirichlet
/// condition.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);
///
/// // u = 0 wherever the first input coordinate is at 0 or 1
/// var bc = new DirichletBoundaryCondition&lt;double&gt;(boundaryValue: 0.0, lowerBound: 0.0, upperBound: 1.0);
///
/// var pinn = new PhysicsInformedNeuralNetwork&lt;double&gt;(
///     architecture, new HeatEquation&lt;double&gt;(), new IBoundaryCondition&lt;double&gt;[] { bc });
/// </code>
/// </example>
public class DirichletBoundaryCondition<T> : IBoundaryCondition<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly T _boundaryValue;
    private readonly T _lowerBound;
    private readonly T _upperBound;
    private readonly T _tolerance;
    private readonly int _coordinate;

    /// <summary>
    /// Initializes a Dirichlet condition on one input coordinate.
    /// </summary>
    /// <param name="boundaryValue">The value the solution is held to on the boundary.</param>
    /// <param name="lowerBound">The coordinate value at the lower edge of the domain.</param>
    /// <param name="upperBound">The coordinate value at the upper edge.</param>
    /// <param name="coordinate">Which input coordinate the boundary is defined on; 0 is usually position.</param>
    /// <param name="tolerance">
    /// How close to an edge counts as being on it. Collocation points are sampled, so they land near the
    /// boundary rather than exactly on it, and an exact comparison would select none of them.
    /// </param>
    public DirichletBoundaryCondition(
        double boundaryValue = 0.0,
        double lowerBound = 0.0,
        double upperBound = 1.0,
        int coordinate = 0,
        double tolerance = 1e-6)
    {
        if (coordinate < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(coordinate), "Coordinate index must be zero or greater.");
        }

        if (tolerance <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tolerance),
                "Tolerance must be positive: sampled collocation points land near a boundary rather " +
                "than exactly on it, so a zero tolerance selects none of them.");
        }

        _boundaryValue = _numOps.FromDouble(boundaryValue);
        _lowerBound = _numOps.FromDouble(lowerBound);
        _upperBound = _numOps.FromDouble(upperBound);
        _tolerance = _numOps.FromDouble(tolerance);
        _coordinate = coordinate;
    }

    /// <inheritdoc/>
    public string Name => "Dirichlet";

    /// <inheritdoc/>
    public bool IsOnBoundary(Vector<T> inputs)
    {
        if (inputs is null || inputs.Length <= _coordinate)
        {
            return false;
        }

        T value = inputs[_coordinate];
        return Within(value, _lowerBound) || Within(value, _upperBound);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// How far the solution is from where it is pinned. The trainer squares and averages these, so the
    /// sign does not matter but the magnitude does.
    /// </remarks>
    public T ComputeBoundaryResidual(
        Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
    {
        if (outputs is null || outputs.Length == 0)
        {
            return _numOps.Zero;
        }

        return _numOps.Subtract(outputs[0], _boundaryValue);
    }

    private bool Within(T value, T edge) =>
        _numOps.LessThanOrEquals(_numOps.Abs(_numOps.Subtract(value, edge)), _tolerance);
}

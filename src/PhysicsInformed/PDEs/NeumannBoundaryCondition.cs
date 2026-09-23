using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.PDEs;

/// <summary>
/// Pins the solution's slope on the boundary rather than its value: <c>∂u/∂n = flux</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The other half of what a boundary-value problem usually needs, alongside
/// <see cref="DirichletBoundaryCondition{T}"/> (#2099).
/// </para>
/// <para>
/// <b>For Beginners:</b> A Dirichlet condition says what the value is at the edge. A Neumann condition
/// says how fast it is changing there — which is what you know when the edge is insulated, or when a
/// fixed amount is flowing in. An insulated rod end has zero flux: no heat crosses it, so the
/// temperature gradient there is zero. That is the default here.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);
///
/// // Insulated at both ends: no flux across the boundary
/// var insulated = new NeumannBoundaryCondition&lt;double&gt;(flux: 0.0, lowerBound: 0.0, upperBound: 1.0);
///
/// var pinn = new PhysicsInformedNeuralNetwork&lt;double&gt;(
///     architecture, new HeatEquation&lt;double&gt;(), new IBoundaryCondition&lt;double&gt;[] { insulated });
/// </code>
/// </example>
public class NeumannBoundaryCondition<T> : IBoundaryCondition<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly T _flux;
    private readonly T _lowerBound;
    private readonly T _upperBound;
    private readonly T _tolerance;
    private readonly int _coordinate;

    /// <summary>
    /// Initializes a Neumann condition on one input coordinate.
    /// </summary>
    /// <param name="flux">The prescribed derivative at the boundary. Zero means insulated.</param>
    /// <param name="lowerBound">The coordinate value at the lower edge of the domain.</param>
    /// <param name="upperBound">The coordinate value at the upper edge.</param>
    /// <param name="coordinate">Which input coordinate the boundary is defined on; 0 is usually position.</param>
    /// <param name="tolerance">
    /// How close to an edge counts as being on it. Collocation points are sampled, so they land near the
    /// boundary rather than exactly on it, and an exact comparison would select none of them.
    /// </param>
    public NeumannBoundaryCondition(
        double flux = 0.0,
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

        _flux = _numOps.FromDouble(flux);
        _lowerBound = _numOps.FromDouble(lowerBound);
        _upperBound = _numOps.FromDouble(upperBound);
        _tolerance = _numOps.FromDouble(tolerance);
        _coordinate = coordinate;
    }

    /// <inheritdoc/>
    public string Name => "Neumann";

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
    /// How far the solution's slope is from the prescribed flux. Without first derivatives there is
    /// nothing to compare, so this reports no violation rather than inventing one.
    /// </remarks>
    public T ComputeBoundaryResidual(
        Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
    {
        if (derivatives?.FirstDerivatives is null)
        {
            return _numOps.Zero;
        }

        T slope = derivatives.FirstDerivatives[0, _coordinate];
        return _numOps.Subtract(slope, _flux);
    }

    private bool Within(T value, T edge) =>
        _numOps.LessThanOrEquals(_numOps.Abs(_numOps.Subtract(value, edge)), _tolerance);
}

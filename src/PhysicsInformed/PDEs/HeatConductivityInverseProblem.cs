using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs;

/// <summary>
/// Recovers an unknown thermal conductivity from measurements: given where and when you probed, and what
/// you read, find the coefficient that explains it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the reference <see cref="IInverseProblem{T}"/>, and until it existed there was none:
/// <c>InverseProblemPINN</c> takes one in its constructor and the library implemented the interface zero
/// times, so the type could not be constructed at all (#2105).
/// </para>
/// <para>
/// <b>For Beginners:</b> A forward problem asks "given the conductivity, what temperatures do I see?".
/// An inverse problem runs it backwards: you have the thermometer readings and want the conductivity,
/// which is the thing you could not measure directly. That is most of experimental science.
/// </para>
/// <para>
/// The forward model is the shipped <see cref="HeatEquation{T}"/> with the current parameter guess
/// substituted in, so the search is over one number: the conductivity itself.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);
///
/// // What you measured: a temperature at each (x, t) you probed.
/// var observations = new[]
/// {
///     (location: new[] { 0.25, 0.1 }, value: new[] { 0.82 }),
///     (location: new[] { 0.50, 0.1 }, value: new[] { 0.95 }),
///     (location: new[] { 0.75, 0.1 }, value: new[] { 0.80 })
/// };
///
/// var problem = new HeatConductivityInverseProblem&lt;double&gt;(observations, initialGuess: 0.5);
/// var pinn = new InverseProblemPINN&lt;double&gt;(
///     architecture, problem, new IBoundaryCondition&lt;double&gt;[] { new DirichletBoundaryCondition&lt;double&gt;() });
/// </code>
/// </example>
public class HeatConductivityInverseProblem<T> : IInverseProblem<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly IReadOnlyList<(T[] location, T[] value)> _observations;
    private readonly T _initialGuess;
    private readonly T _lowerBound;
    private readonly T _upperBound;
    private readonly T _noiseLevel;
    private readonly bool _hasNoiseLevel;

    /// <summary>
    /// Initializes the problem from what was measured.
    /// </summary>
    /// <param name="observations">Each measurement: where it was taken, and what was read.</param>
    /// <param name="initialGuess">Where the search for the conductivity starts.</param>
    /// <param name="lowerBound">The smallest conductivity considered physical.</param>
    /// <param name="upperBound">The largest.</param>
    /// <param name="measurementNoiseLevel">
    /// The standard deviation of your instrument, when you know it. Leave null when you do not: an
    /// invented noise level is worse than none, because it tells the fit how much to trust the data.
    /// </param>
    public HeatConductivityInverseProblem(
        IEnumerable<(double[] location, double[] value)> observations,
        double initialGuess = 1.0,
        double lowerBound = 0.001,
        double upperBound = 100.0,
        double? measurementNoiseLevel = null)
    {
        if (observations is null) throw new ArgumentNullException(nameof(observations));

        if (lowerBound <= 0 || upperBound <= lowerBound)
        {
            throw new ArgumentException(
                $"Conductivity bounds must satisfy 0 < lower < upper; got [{lowerBound}, {upperBound}]. " +
                "A non-positive conductivity is not physical and makes the forward equation degenerate.",
                nameof(lowerBound));
        }

        if (initialGuess < lowerBound || initialGuess > upperBound)
        {
            throw new ArgumentOutOfRangeException(
                nameof(initialGuess),
                $"The starting guess {initialGuess} is outside the bounds [{lowerBound}, {upperBound}], " +
                "so the search would begin somewhere it is not allowed to be.");
        }

        _observations = observations
            .Select(o => (
                o.location.Select(_numOps.FromDouble).ToArray(),
                o.value.Select(_numOps.FromDouble).ToArray()))
            .ToList();

        if (_observations.Count == 0)
        {
            throw new ArgumentException(
                "An inverse problem needs at least one observation — the measurements are the only " +
                "thing distinguishing it from the forward problem.",
                nameof(observations));
        }

        _initialGuess = _numOps.FromDouble(initialGuess);
        _lowerBound = _numOps.FromDouble(lowerBound);
        _upperBound = _numOps.FromDouble(upperBound);
        _hasNoiseLevel = measurementNoiseLevel.HasValue;
        _noiseLevel = _numOps.FromDouble(measurementNoiseLevel ?? 0.0);
    }

    /// <inheritdoc/>
    public string[] ParameterNames => ["thermal_conductivity"];

    /// <inheritdoc/>
    public int NumberOfParameters => 1;

    /// <inheritdoc/>
    public T[] InitialParameterGuesses => [_initialGuess];

    /// <inheritdoc/>
    public T[]? ParameterLowerBounds => [_lowerBound];

    /// <inheritdoc/>
    public T[]? ParameterUpperBounds => [_upperBound];

    /// <inheritdoc/>
    public IReadOnlyList<(T[] location, T[] value)> Observations => _observations;

    /// <inheritdoc/>
    public bool HasMeasurementNoiseLevel => _hasNoiseLevel;

    /// <inheritdoc/>
    public T MeasurementNoiseLevel => _noiseLevel;

    /// <inheritdoc/>
    public bool ValidateParameters(T[] parameters)
    {
        if (parameters is null || parameters.Length != NumberOfParameters)
        {
            return false;
        }

        T conductivity = parameters[0];
        return _numOps.GreaterThanOrEquals(conductivity, _lowerBound)
            && _numOps.LessThanOrEquals(conductivity, _upperBound);
    }

    /// <inheritdoc/>
    /// <remarks>The forward equation with the current guess substituted in.</remarks>
    public IPDESpecification<T> CreateParameterizedPDE(T[] parameters)
    {
        if (!ValidateParameters(parameters))
        {
            throw new ArgumentException(
                "Parameters are outside the bounds this problem was given, so the forward equation " +
                "cannot be built from them.",
                nameof(parameters));
        }

        return new HeatEquation<T>(parameters[0]);
    }
}

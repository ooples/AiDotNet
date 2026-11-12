namespace AiDotNet.Kernels;

/// <summary>
/// Cylindrical Kernel for Bayesian optimization with periodic/angular dimensions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Cylindrical Kernel is designed for data that has both
/// "regular" dimensions (like temperature, pressure) and "angular" or "periodic"
/// dimensions (like angles, time of day, day of week).
///
/// For regular dimensions: Uses a standard kernel (e.g., RBF)
/// For angular dimensions: Uses a periodic kernel that wraps around
///
/// Example use cases:
/// - Optimizing chemical reactions: Temperature (regular) + catalyst angle (periodic)
/// - Time series: Date (regular) + hour of day (periodic)
/// - Robotics: Position (regular) + joint angles (periodic)
///
/// The kernel combines:
/// k(x, x') = k_regular(x_reg, x'_reg) × k_angular(x_ang, x'_ang)
///
/// For angular dimensions, it uses:
/// k_angular(θ, θ') = exp(-2 × sin²(π(θ - θ')/period) / l²)
///
/// This ensures smooth wrapping at boundaries (e.g., 359° is close to 1°).
/// </para>
/// </remarks>
public class CylindricalKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The kernel for regular (non-periodic) dimensions.
    /// </summary>
    private readonly IKernelFunction<T> _regularKernel;

    /// <summary>
    /// Indices of regular dimensions.
    /// </summary>
    private readonly int[] _regularDims;

    /// <summary>
    /// Indices of angular/periodic dimensions.
    /// </summary>
    private readonly int[] _angularDims;

    /// <summary>
    /// Periods for each angular dimension.
    /// </summary>
    private readonly Vector<T> _periods;

    /// <summary>
    /// Length scales for angular dimensions.
    /// </summary>
    private readonly Vector<T> _angularLengthScales;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Cylindrical Kernel.
    /// </summary>
    /// <param name="regularKernel">Kernel for regular dimensions.</param>
    /// <param name="regularDims">Indices of regular dimensions.</param>
    /// <param name="angularDims">Indices of angular dimensions.</param>
    /// <param name="periods">Period for each angular dimension. If null, uses 2π for all.</param>
    /// <param name="angularLengthScales">Length scale for each angular dimension. If null, uses 1.0 for all.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a cylindrical kernel with specified dimension types.
    ///
    /// Example for 4D input where dims 0,1 are regular and dims 2,3 are angular:
    /// var rbf = new GaussianKernel&lt;double&gt;(1.0);
    /// var cylindrical = new CylindricalKernel&lt;double&gt;(
    ///     rbf,
    ///     regularDims: new[] { 0, 1 },
    ///     angularDims: new[] { 2, 3 },
    ///     periods: new Vector&lt;double&gt;(new[] { 2 * Math.PI, 24.0 })  // Angles in radians, hours in day
    /// );
    /// </para>
    /// </remarks>
    public CylindricalKernel(
        IKernelFunction<T> regularKernel,
        int[] regularDims,
        int[] angularDims,
        Vector<T>? periods = null,
        Vector<T>? angularLengthScales = null)
    {
        if (regularKernel is null) throw new ArgumentNullException(nameof(regularKernel));
        if (regularDims is null) throw new ArgumentNullException(nameof(regularDims));
        if (angularDims is null) throw new ArgumentNullException(nameof(angularDims));
        if (angularDims.Length == 0)
            throw new ArgumentException("Must have at least one angular dimension.", nameof(angularDims));

        _regularKernel = regularKernel;
        _regularDims = (int[])regularDims.Clone();
        _angularDims = (int[])angularDims.Clone();

        _numOps = MathHelper.GetNumericOperations<T>();

        // Set up periods
        if (periods is null)
        {
            _periods = Vector<T>.CreateDefault(angularDims.Length, _numOps.FromDouble(2 * Math.PI));
        }
        else
        {
            if (periods.Length != angularDims.Length)
                throw new ArgumentException("Periods length must match angular dimensions count.");
            _periods = periods.Clone();
        }

        // Set up length scales
        if (angularLengthScales is null)
        {
            _angularLengthScales = Vector<T>.CreateDefault(angularDims.Length, _numOps.One);
        }
        else
        {
            if (angularLengthScales.Length != angularDims.Length)
                throw new ArgumentException("Angular length scales length must match angular dimensions count.");
            _angularLengthScales = angularLengthScales.Clone();
        }

        // Validate
        var engine = AiDotNetEngine.Current;
        if (_numOps.LessThanOrEquals(engine.TensorMinValue(_periods.AsTensor()), _numOps.Zero))
            throw new ArgumentException("Periods must be positive.");
        if (_numOps.LessThanOrEquals(engine.TensorMinValue(_angularLengthScales.AsTensor()), _numOps.Zero))
            throw new ArgumentException("Length scales must be positive.");
    }

    /// <summary>
    /// Gets the regular kernel.
    /// </summary>
    public IKernelFunction<T> RegularKernel => _regularKernel;

    /// <summary>
    /// Gets the regular dimension indices.
    /// </summary>
    public int[] RegularDims => (int[])_regularDims.Clone();

    /// <summary>
    /// Gets the angular dimension indices.
    /// </summary>
    public int[] AngularDims => (int[])_angularDims.Clone();

    /// <summary>
    /// Calculates the cylindrical kernel value.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value combining regular and angular components.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel value is the product of:
    /// 1. Regular kernel applied to regular dimensions
    /// 2. Periodic kernels applied to each angular dimension
    ///
    /// This ensures proper wrapping behavior for angular dimensions.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T result = _numOps.One;

        // Compute regular kernel if there are regular dimensions
        if (_regularDims.Length > 0)
        {
            var reg1 = ExtractDims(x1, _regularDims);
            var reg2 = ExtractDims(x2, _regularDims);
            result = _regularKernel.Calculate(reg1, reg2);
        }

        // Compute periodic kernel for each angular dimension
        // Periodic kernel per angular dimension: exp(-2 * sin^2(pi * (v1 - v2) / period) / l^2)
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(ExtractDims(x1, _angularDims), ExtractDims(x2, _angularDims));
        var sinVal = engine.Sin(engine.Divide(engine.Multiply(diff, _numOps.FromDouble(Math.PI)), _periods));
        var exponent = engine.Divide(engine.Multiply(engine.Multiply(sinVal, sinVal), _numOps.FromDouble(-2.0)),
            engine.Multiply(_angularLengthScales, _angularLengthScales));

        return _numOps.Multiply(result, engine.Product(engine.Exp(exponent)));
    }

    /// <summary>
    /// Extracts specified dimensions from a vector.
    /// </summary>
    private Vector<T> ExtractDims(Vector<T> x, int[] dims)
    {
        return x.GetElements(dims);
    }

    /// <summary>
    /// Creates a Cylindrical Kernel with RBF for regular dimensions.
    /// </summary>
    /// <param name="totalDims">Total number of dimensions.</param>
    /// <param name="angularDimIndices">Indices of angular dimensions (rest are regular).</param>
    /// <param name="regularLengthScale">Length scale for RBF kernel on regular dims.</param>
    /// <param name="periods">Periods for angular dimensions.</param>
    /// <param name="angularLengthScales">Length scales for angular dimensions.</param>
    /// <returns>A new Cylindrical Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience factory method that automatically determines
    /// which dimensions are regular based on which are angular.
    /// </para>
    /// </remarks>
    public static CylindricalKernel<T> WithRBF(
        int totalDims,
        int[] angularDimIndices,
        double regularLengthScale = 1.0,
        Vector<T>? periods = null,
        Vector<T>? angularLengthScales = null)
    {
        // Determine regular dimensions (all dims not in angular)
        var angularSet = new HashSet<int>(angularDimIndices);
        var regularDims = new List<int>();
        for (int i = 0; i < totalDims; i++)
        {
            if (!angularSet.Contains(i))
            {
                regularDims.Add(i);
            }
        }

        var rbf = new GaussianKernel<T>(regularLengthScale);
        return new CylindricalKernel<T>(
            rbf,
            regularDims.ToArray(),
            angularDimIndices,
            periods,
            angularLengthScales);
    }
}

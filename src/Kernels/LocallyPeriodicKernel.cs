namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Locally Periodic kernel for measuring similarity between data points with periodic patterns.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Locally Periodic kernel combines periodic behavior with local smoothness, making it ideal for
/// modeling data that shows repeating patterns that may change or decay over time or space.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Locally Periodic kernel is specially designed for data that repeats in patterns (like seasonal sales,
/// daily temperature cycles, or sound waves) but where the pattern might change gradually over time.
/// </para>
/// <para>
/// Think of this kernel as a "pattern detector" that recognizes when two points are at the same phase
/// of a repeating cycle. For example, in temperature data, it would recognize that 3 PM on one day is
/// similar to 3 PM on another day (because they're at the same point in the daily cycle), but this
/// similarity would decrease if the days are far apart (like comparing summer to winter).
/// </para>
/// <para>
/// This kernel has three important parameters:
/// - The period controls how long each cycle is (like 24 hours for daily patterns)
/// - The length scale controls how quickly the similarity decays over multiple cycles
/// - The amplitude controls the overall strength of the pattern
/// </para>
/// <para>
/// The Locally Periodic kernel is particularly useful for time series forecasting, signal processing,
/// and any application where you need to model repeating patterns that evolve over time.
/// </para>
/// </remarks>
public class LocallyPeriodicKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Controls how quickly the similarity decays as points get farther apart in multiple cycles.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of the '_lengthScale' parameter as a "memory span" for the pattern.
    /// 
    /// A larger value means the kernel "remembers" the pattern over a longer distance or time,
    /// so even points that are many cycles apart can still be considered similar if they're at
    /// the same phase of their respective cycles.
    /// 
    /// A smaller value means the kernel has a "shorter memory" - it only considers points to be
    /// similar if they're at the same phase AND relatively close to each other in time or space.
    /// 
    /// The default value is 1.0, which provides a balanced approach for many applications.
    /// </remarks>
    private readonly T _lengthScale;

    /// <summary>
    /// Defines the length of one complete cycle in the periodic pattern.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_period' parameter tells the kernel how long each repeating cycle is.
    /// 
    /// For example:
    /// - If you're analyzing daily temperature patterns, the period might be 24 hours
    /// - If you're analyzing yearly sales data, the period might be 12 months
    /// - If you're analyzing sound waves, the period would be the wavelength of the sound
    /// 
    /// The default value is 2p (approximately 6.28), which is the standard period for trigonometric functions.
    /// You'll typically want to set this to match the natural cycle length in your data.
    /// </remarks>
    private readonly T _period;

    /// <summary>
    /// Controls the overall strength or magnitude of the kernel output.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The '_amplitude' parameter controls how "strong" the pattern signal is.
    /// 
    /// Think of it as a volume knob - a higher amplitude means the periodic pattern has a stronger
    /// influence on your model's predictions. A lower amplitude means the pattern has a more subtle effect.
    /// 
    /// The default value is 1.0, which is a standard starting point. You might increase this if the
    /// periodic patterns in your data are very pronounced, or decrease it if they're more subtle.
    /// </remarks>
    private readonly T _amplitude;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Locally Periodic kernel with optional parameters.
    /// </summary>
    /// <param name="lengthScale">Controls how quickly similarity decays over multiple cycles. Default is 1.0.</param>
    /// <param name="period">Defines the length of one complete cycle. Default is 2p (approximately 6.28).</param>
    /// <param name="amplitude">Controls the overall strength of the pattern. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Locally Periodic kernel for use. You can optionally
    /// provide values for the three parameters that control how the kernel behaves.
    /// </para>
    /// <para>
    /// If you don't specify values, the defaults are:
    /// - lengthScale = 1.0: A standard value for how quickly similarity decays over distance
    /// - period = 2p (approximately 6.28): The standard period for trigonometric functions
    /// - amplitude = 1.0: A standard strength for the pattern
    /// </para>
    /// <para>
    /// When might you want to change these parameters?
    /// - Change the lengthScale if you want to control how far the pattern "remembers" itself
    /// - Change the period to match the natural cycle length in your data (daily, yearly, etc.)
    /// - Change the amplitude if you want to make the periodic pattern more or less influential
    /// </para>
    /// <para>
    /// The best values for these parameters often depend on your specific dataset and problem,
    /// so you might need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public LocallyPeriodicKernel(T? lengthScale = default, T? period = default, T? amplitude = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _lengthScale = lengthScale ?? _numOps.One;
        _period = period ?? _numOps.FromDouble(2 * Math.PI);
        _amplitude = amplitude ?? _numOps.One;
    }

    /// <summary>
    /// Calculates the Locally Periodic kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Locally Periodic kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the distance between the two points
    /// 2. Calculating how far apart they are in terms of the periodic cycle
    /// 3. Applying a decay factor based on how many cycles apart they are
    /// 4. Scaling the result by the amplitude
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A higher value means the points are at similar phases in their cycles and not too far apart
    /// - A lower value means the points are either at different phases or very far apart
    /// </para>
    /// <para>
    /// This kernel is particularly good at finding patterns like:
    /// - Daily temperature cycles that change with the seasons
    /// - Yearly sales patterns that evolve over multiple years
    /// - Heartbeat signals that may change their rhythm over time
    /// </para>
    /// <para>
    /// The Locally Periodic kernel combines the best of both worlds: it can detect repeating patterns
    /// like the Periodic kernel, but it also accounts for how these patterns might change over time
    /// or space, making it more flexible for real-world data.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = x1.EuclideanDistance(x2);
        T periodicTerm = MathHelper.Sin(_numOps.Divide(_numOps.Multiply(MathHelper.Pi<T>(), distance), _period));
        T squaredPeriodicTerm = _numOps.Square(periodicTerm);
        T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), squaredPeriodicTerm), _numOps.Square(_lengthScale))));

        return _numOps.Multiply(_numOps.Square(_amplitude), expTerm);
    }
}

using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Abstract base class for synthetic tabular data generators, providing common infrastructure
/// for fitting models on real data and generating synthetic rows.
/// </summary>
/// <remarks>
/// <para>
/// This base class handles the lifecycle shared by all tabular generators:
/// column metadata management, random number generation, fitted-state tracking,
/// and the public Fit/Generate API that delegates to subclass implementations.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the shared foundation that CTGAN, TVAE, and TabDDPM build on.
/// It handles the "bookkeeping" that every generator needs:
/// - Remembering column descriptions (which columns are numbers vs categories)
/// - Managing the random number generator (for reproducibility)
/// - Tracking whether the model has been trained yet
/// - Computing basic statistics (min, max, mean, std) for each column
///
/// Specific generators override <see cref="FitInternal"/> and <see cref="GenerateInternal"/>
/// to implement their unique training and generation algorithms.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
public abstract class SyntheticTabularGeneratorBase<T> : ISyntheticTabularGenerator<T>, IJitCompilable<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected Random Random;

    /// <summary>
    /// The stored column metadata after fitting.
    /// </summary>
    private List<ColumnMetadata> _columns = new();

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the base class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility. If null, uses cryptographically secure random.</param>
    protected SyntheticTabularGeneratorBase(int? seed = null)
    {
        Random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        // Clone and index columns, then compute statistics
        _columns = PrepareColumns(data, columns);

        // Delegate to subclass implementation
        FitInternal(data, _columns, epochs);

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            FitInternal(data, _columns, epochs);
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        return GenerateInternal(numSamples, conditionColumn, conditionValue);
    }

    /// <summary>
    /// Subclass-specific training implementation.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Prepared column metadata with computed statistics.</param>
    /// <param name="epochs">Number of training epochs.</param>
    protected abstract void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs);

    /// <summary>
    /// Subclass-specific generation implementation.
    /// </summary>
    /// <param name="numSamples">Number of rows to generate.</param>
    /// <param name="conditionColumn">Optional conditioning column indices.</param>
    /// <param name="conditionValue">Optional conditioning values.</param>
    /// <returns>A matrix of synthetic data.</returns>
    protected abstract Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue);

    /// <summary>
    /// Validates inputs to the Fit method.
    /// </summary>
    private static void ValidateFitInputs(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        if (data.Rows == 0 || data.Columns == 0)
        {
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        }

        if (columns.Count == 0)
        {
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        }

        if (columns.Count != data.Columns)
        {
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).",
                nameof(columns));
        }

        if (epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        }
    }

    /// <summary>
    /// Clones column metadata, assigns indices, and computes statistics from data.
    /// </summary>
    private List<ColumnMetadata> PrepareColumns(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        var prepared = new List<ColumnMetadata>(columns.Count);

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col].Clone();
            meta.ColumnIndex = col;

            if (meta.IsNumerical)
            {
                ComputeColumnStatistics(data, col, meta);
            }
            else if (meta.IsCategorical && meta.NumCategories == 0)
            {
                // Auto-detect categories from data
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                {
                    var val = NumOps.ToDouble(data[row, col]);
                    categories.Add(val.ToString(CultureInfo.InvariantCulture));
                }
                meta.Categories = categories.OrderBy(c => c, StringComparer.Ordinal).ToList().AsReadOnly();
            }

            prepared.Add(meta);
        }

        return prepared;
    }

    /// <summary>
    /// Computes min, max, mean, and standard deviation for a numerical column.
    /// </summary>
    private static void ComputeColumnStatistics(Matrix<T> data, int colIndex, ColumnMetadata meta)
    {
        int n = data.Rows;
        double sum = 0;
        double min = double.MaxValue;
        double max = double.MinValue;

        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            sum += val;
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double mean = sum / n;
        double sumSqDiff = 0;
        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            double diff = val - mean;
            sumSqDiff += diff * diff;
        }

        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10; // Prevent division by zero

        meta.Min = min;
        meta.Max = max;
        meta.Mean = mean;
        meta.Std = std;
    }

    #region Gradient Safety Utilities

    /// <summary>
    /// Clips a gradient tensor to a maximum L2 norm, preventing exploding gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, gradients can sometimes become very large (explode),
    /// causing the model to diverge. Gradient clipping rescales the gradient if its magnitude
    /// exceeds a threshold, keeping training stable.
    /// </para>
    /// </remarks>
    /// <param name="grad">The gradient tensor to clip.</param>
    /// <param name="maxNorm">The maximum allowed L2 norm. Typical values: 1.0 to 10.0.</param>
    /// <returns>The clipped gradient tensor.</returns>
    protected static Tensor<T> ClipGradientNorm(Tensor<T> grad, double maxNorm)
    {
        if (maxNorm <= 0) return grad;

        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            normSq += v * v;
        }

        double norm = Math.Sqrt(normSq);
        if (norm <= maxNorm) return grad;

        double scale = maxNorm / norm;
        var clipped = new Tensor<T>(grad.Shape);
        for (int i = 0; i < grad.Length; i++)
        {
            clipped[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
        }
        return clipped;
    }

    /// <summary>
    /// Checks whether a tensor contains any NaN or Infinity values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NaN (Not a Number) values can appear during training when computations
    /// go wrong (e.g., dividing by zero, taking log of negative numbers). Once a NaN appears,
    /// it propagates through all subsequent computations, ruining the model. Detecting NaN early
    /// allows corrective action.
    /// </para>
    /// </remarks>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if the tensor contains NaN or Infinity; false otherwise.</returns>
    protected static bool HasNaN(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Replaces NaN and Infinity values in a tensor with zero, in-place.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When NaN values are detected in gradients, replacing them with zero
    /// effectively skips that gradient update. This is safer than letting NaN propagate through
    /// the entire model and corrupt all parameters.
    /// </para>
    /// </remarks>
    /// <param name="tensor">The tensor to sanitize in-place.</param>
    protected static void SanitizeTensor(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                tensor[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Applies NaN sanitization and gradient norm clipping in a single operation.
    /// This is the recommended method to call on all gradients before using them for backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines two safety checks into one convenient call:
    /// 1. Replace any NaN/Infinity values with zero
    /// 2. If the gradient is too large, scale it down to a safe magnitude
    ///
    /// Call this on every gradient tensor before passing it to a backward pass for maximum training stability.
    /// </para>
    /// </remarks>
    /// <param name="grad">The gradient tensor to sanitize and clip.</param>
    /// <param name="maxNorm">The maximum allowed L2 norm. Default is 1.0.</param>
    /// <returns>A safe, clipped gradient tensor.</returns>
    protected static Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm = 1.0)
    {
        SanitizeTensor(grad);
        return ClipGradientNorm(grad, maxNorm);
    }

    /// <summary>
    /// Checks a vector for NaN/Infinity values, returning true if any are found.
    /// </summary>
    /// <param name="vector">The vector to check.</param>
    /// <returns>True if NaN or Infinity is found.</returns>
    protected static bool HasNaN(Vector<T> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            double v = NumOps.ToDouble(vector[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Replaces NaN and Infinity values in a vector with zero, in-place.
    /// </summary>
    /// <param name="vector">The vector to sanitize.</param>
    protected static void SanitizeTensor(Vector<T> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            double v = NumOps.ToDouble(vector[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                vector[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Clips a gradient vector to a maximum L2 norm, preventing exploding gradients.
    /// </summary>
    /// <param name="grad">The gradient vector to clip.</param>
    /// <param name="maxNorm">The maximum allowed L2 norm.</param>
    /// <returns>The clipped gradient vector.</returns>
    protected static Vector<T> ClipGradientNorm(Vector<T> grad, double maxNorm)
    {
        if (maxNorm <= 0) return grad;

        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            normSq += v * v;
        }

        double norm = Math.Sqrt(normSq);
        if (norm <= maxNorm) return grad;

        double scale = maxNorm / norm;
        var clipped = new Vector<T>(grad.Length);
        for (int i = 0; i < grad.Length; i++)
        {
            clipped[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
        }
        return clipped;
    }

    /// <summary>
    /// Applies NaN sanitization and gradient norm clipping to a vector in a single operation.
    /// This is the recommended method to call on all gradient vectors before using them for backward passes.
    /// </summary>
    /// <param name="grad">The gradient vector to sanitize and clip.</param>
    /// <param name="maxNorm">The maximum allowed L2 norm. Default is 1.0.</param>
    /// <returns>A safe, clipped gradient vector.</returns>
    protected static Vector<T> SafeGradient(Vector<T> grad, double maxNorm = 1.0)
    {
        SanitizeTensor(grad);
        return ClipGradientNorm(grad, maxNorm);
    }

    /// <summary>
    /// Checks whether a loss value is valid (not NaN or Infinity).
    /// Returns true if the loss is finite, false otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, the loss value should always be a finite number.
    /// If it becomes NaN (Not a Number) or Infinity, something has gone wrong — typically
    /// due to numerical instability like dividing by zero or taking log of a negative number.
    /// Use this method to detect divergence early and skip or reset problematic training steps.
    /// </para>
    /// </remarks>
    /// <param name="loss">The loss value to check.</param>
    /// <returns>True if the loss is a valid finite number; false if NaN or Infinity.</returns>
    protected static bool IsFiniteLoss(double loss)
    {
        return !double.IsNaN(loss) && !double.IsInfinity(loss);
    }

    #endregion

    /// <summary>
    /// Samples a standard normal random value as type T.
    /// </summary>
    /// <returns>A random value drawn from N(0, 1).</returns>
    protected T SampleStandardNormal()
    {
        // Box-Muller transform
        double u1 = 1.0 - Random.NextDouble(); // Avoid log(0)
        double u2 = Random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    /// <summary>
    /// Fills a vector with standard normal random values.
    /// </summary>
    /// <param name="vector">The vector to fill.</param>
    protected void FillStandardNormal(Vector<T> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = SampleStandardNormal();
        }
    }

    /// <summary>
    /// Creates a vector filled with standard normal random values.
    /// </summary>
    /// <param name="length">The vector length.</param>
    /// <returns>A new vector of standard normal random values.</returns>
    protected Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        FillStandardNormal(v);
        return v;
    }

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this generator supports JIT compilation for accelerated generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Generators with simple MLP-based architectures (GAN generators) can export their
    /// generator network as a computation graph for JIT compilation, which accelerates
    /// the neural network forward pass during synthetic data generation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> JIT compilation converts the generator's neural network into
    /// optimized native code. This makes the Generate() method faster, especially when
    /// generating large numbers of synthetic rows. Generators based on diffusion models,
    /// autoregressive transformers, or statistical methods typically cannot be JIT compiled
    /// because they use iterative or dynamic computation patterns.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Exports the generator network's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes (the noise vector).</param>
    /// <returns>The output computation node representing the generator's raw output.</returns>
    /// <remarks>
    /// <para>
    /// Subclasses that support JIT compilation should override this method to export their
    /// generator network as a computation graph using TensorOperations. The exported graph
    /// represents the forward pass from noise input to raw generated output (before inverse
    /// data transformation).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a "recipe" of the generator's calculations that
    /// the JIT compiler can optimize into fast native code. Only the neural network portion
    /// is exported — the data transformation and post-processing steps remain interpreted.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown when the generator does not support JIT compilation.</exception>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support JIT compilation. Check SupportsJitCompilation before calling.");
    }

    /// <summary>
    /// Exports a sequential MLP generator as a computation graph, with optional residual concatenation.
    /// Delegates to <see cref="TapeLayerBridge{T}.ExportMLPGeneratorGraph"/> for the implementation.
    /// </summary>
    protected static ComputationNode<T> ExportMLPGeneratorGraph(
        List<ComputationNode<T>> inputNodes,
        int inputSize,
        IReadOnlyList<ILayer<T>> hiddenLayers,
        IReadOnlyList<ILayer<T>>? bnLayers,
        ILayer<T> outputLayer,
        TapeLayerBridge<T>.HiddenActivation hiddenAct = TapeLayerBridge<T>.HiddenActivation.ReLU,
        TapeLayerBridge<T>.HiddenActivation outputAct = TapeLayerBridge<T>.HiddenActivation.None,
        bool useResidualConcat = false)
    {
        return TapeLayerBridge<T>.ExportMLPGeneratorGraph(
            inputNodes, inputSize, hiddenLayers, bnLayers, outputLayer,
            hiddenAct, outputAct, useResidualConcat);
    }

    #endregion
}

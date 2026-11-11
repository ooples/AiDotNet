using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Provides GPU acceleration for interpretability explainers.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper accelerates interpretability computations using GPU hardware.
/// Many explanation methods (SHAP, Integrated Gradients, etc.) require computing many predictions
/// or gradients - operations that are perfect for GPU parallelization.
///
/// The helper provides:
/// 1. <b>Batch Prediction</b>: Process many inputs simultaneously on GPU
/// 2. <b>Parallel Coalition Processing</b>: For SHAP-style algorithms
/// 3. <b>GPU Matrix Operations</b>: Fast linear algebra for solving attribution problems
/// 4. <b>Automatic Fallback</b>: Falls back to CPU if no GPU is available
///
/// Benefits of GPU acceleration for explainers:
/// - 10-100x speedup for batch predictions
/// - Enables real-time explanations for complex models
/// - Makes ensemble/sample-based methods practical for large models
///
/// Usage:
/// <code>
/// var gpuHelper = new GPUExplainerHelper&lt;double&gt;(gpuRuntime);
/// // Or auto-detect GPU
/// var gpuHelper2 = GPUExplainerHelper&lt;double&gt;.CreateWithAutoDetect();
///
/// // Batch process predictions
/// var predictions = gpuHelper2.BatchPredict(model, inputs);
///
/// // Parallel coalition processing for SHAP
/// var values = gpuHelper2.ComputeCoalitionPredictions(model, instance, coalitions, background);
/// </code>
/// </para>
/// </remarks>
public class GPUExplainerHelper<T> : IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _maxParallelism;
    private bool _disposed;

    /// <summary>
    /// Gets whether GPU acceleration is available and enabled.
    /// Always returns false — per-layer GPU support was removed with the JIT compiler.
    /// GPU acceleration is now handled at the Tensors engine level.
    /// </summary>
    public bool IsGPUEnabled => false;

    /// <summary>
    /// Gets information about the GPU device.
    /// Always returns null — per-layer GPU support was removed with the JIT compiler.
    /// GPU device management is now handled at the Tensors engine level.
    /// </summary>
    public object? DeviceInfo => null;

    /// <summary>
    /// Gets the maximum parallelism level for CPU fallback operations.
    /// </summary>
    public int MaxParallelism => _maxParallelism;

    /// <summary>
    /// Initializes a new GPU explainer helper using CPU parallel processing.
    /// </summary>
    /// <param name="maxParallelism">Maximum parallelism for CPU operations (default: processor count).</param>
    /// <remarks>
    /// Per-layer GPU support was removed with the JIT compiler. GPU acceleration
    /// is now handled at the Tensors engine level. This helper uses CPU parallelism.
    /// </remarks>
    public GPUExplainerHelper(int? maxParallelism = null)
    {
        var resolved = maxParallelism ?? Environment.ProcessorCount;
        if (resolved <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxParallelism), resolved, "Must be at least 1.");
        _maxParallelism = resolved;
    }

    /// <summary>
    /// Creates a GPU explainer helper with automatic GPU detection.
    /// </summary>
    /// <returns>A helper configured with the best available compute device.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This factory method automatically detects if a GPU is available
    /// and configures the helper appropriately. Use this when you want the best performance
    /// without manually managing GPU resources.
    ///
    /// Currently returns a CPU-based helper (Mock GPU) since actual GPU detection requires
    /// platform-specific code. Future versions may include actual GPU detection.
    /// </para>
    /// </remarks>
    public static GPUExplainerHelper<T> CreateWithAutoDetect()
    {
        // GPU runtime has been removed — returns CPU-only helper
        return new GPUExplainerHelper<T>();
    }

    /// <summary>
    /// Creates a CPU-only helper (no GPU acceleration).
    /// </summary>
    /// <returns>A helper that uses only CPU parallelization.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you explicitly don't want GPU acceleration
    /// or when running on systems without GPU support.
    /// </para>
    /// </remarks>
    public static GPUExplainerHelper<T> CreateCPUOnly(int? maxParallelism = null)
    {
        return new GPUExplainerHelper<T>(maxParallelism);
    }

    /// <summary>
    /// Processes predictions for multiple inputs in parallel.
    /// </summary>
    /// <param name="predictFunction">The prediction function to apply.</param>
    /// <param name="inputs">Matrix of input samples (each row is one sample).</param>
    /// <returns>Vector of predictions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs predictions for many inputs simultaneously.
    /// Instead of processing one input at a time, it batches them for efficiency.
    ///
    /// This is particularly useful for:
    /// - SHAP/LIME which need many masked predictions
    /// - Integrated Gradients which samples along a path
    /// - Permutation importance which shuffles features many times
    ///
    /// The GPU version can process thousands of inputs simultaneously.
    /// The CPU version uses thread parallelism for moderate speedup.
    /// </para>
    /// </remarks>
    public Vector<T> BatchPredict(Func<Matrix<T>, Vector<T>> predictFunction, Matrix<T> inputs)
    {
        if (inputs.Rows == 0)
            return new Vector<T>(0);

        // For small batches, direct computation is faster
        if (inputs.Rows <= 10)
        {
            return predictFunction(inputs);
        }

        // Use parallel batch processing
        if (IsGPUEnabled)
        {
            return BatchPredictGPU(predictFunction, inputs);
        }
        else
        {
            return BatchPredictCPU(predictFunction, inputs);
        }
    }

    /// <summary>
    /// GPU-accelerated batch prediction.
    /// </summary>
    private Vector<T> BatchPredictGPU(Func<Matrix<T>, Vector<T>> predictFunction, Matrix<T> inputs)
    {
        // The GPU runtime processes the full batch in one call
        // This is efficient because data transfer overhead is amortized
        return predictFunction(inputs);
    }

    /// <summary>
    /// CPU parallel batch prediction.
    /// </summary>
    private Vector<T> BatchPredictCPU(Func<Matrix<T>, Vector<T>> predictFunction, Matrix<T> inputs)
    {
        // Determine optimal batch size based on parallelism
        int batchSize = Math.Max(1, inputs.Rows / _maxParallelism);
        int numBatches = (inputs.Rows + batchSize - 1) / batchSize;

        var batches = new List<(int start, int count)>();

        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int count = Math.Min(batchSize, inputs.Rows - start);
            batches.Add((start, count));
        }

        var batchPredictions = new Tensor<T>[batches.Count];

        // Process batches in parallel
        Parallel.For(0, batches.Count, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelism }, b =>
        {
            var batch = batches[b];
            var batchInputs = ExtractRows(inputs, batch.start, batch.count);
            var batchPreds = predictFunction(batchInputs);
            batchPredictions[b] = batchPreds.GetSubVector(0, batch.count).AsTensor();
        });

        if (batchPredictions.Length == 0)
            return new Vector<T>(0);

        return AiDotNetEngine.Current.TensorConcatenate(batchPredictions, 0).ToVector();
    }

    /// <summary>
    /// Computes predictions for multiple coalitions in parallel (for SHAP-style algorithms).
    /// </summary>
    /// <param name="predictFunction">The prediction function.</param>
    /// <param name="instance">The instance being explained.</param>
    /// <param name="coalitions">List of coalitions (boolean arrays indicating which features are included).</param>
    /// <param name="backgroundData">Background data for marginalizing missing features.</param>
    /// <param name="nBackgroundSamples">Number of background samples to use per coalition.</param>
    /// <returns>Vector of expected predictions for each coalition.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In SHAP and similar methods, we need to compute predictions where
    /// some features are "missing" (marginalized out using background data). This involves:
    ///
    /// 1. For each coalition (subset of features)
    /// 2. Create versions of the input where missing features are filled from background data
    /// 3. Average the predictions to get the expected value
    ///
    /// This method parallelizes step 1-3 across all coalitions simultaneously.
    /// For 100 coalitions × 10 background samples = 1000 predictions to process.
    /// GPU acceleration makes this practical even for complex models.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeCoalitionPredictions(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Vector<T> instance,
        List<bool[]> coalitions,
        Matrix<T> backgroundData,
        int nBackgroundSamples = 10)
    {
        int numCoalitions = coalitions.Count;
        int numFeatures = instance.Length;
        int nBackground = Math.Min(backgroundData.Rows, nBackgroundSamples);

        if (numCoalitions == 0)
            return new Vector<T>(0);

        var engine = AiDotNetEngine.Current;
        var background = Tensor<T>.FromMatrix(backgroundData.GetSubMatrix(0, 0, nBackground, numFeatures));
        var instanceRow = instance.AsTensor().Reshape(new[] { 1, numFeatures });

        // Each coalition row: instance value where the feature is in the coalition, background value otherwise.
        bool[] CoalitionMask(IEnumerable<bool[]> selected) =>
            selected.SelectMany(coalition => Enumerable.Repeat(coalition, nBackground))
                .SelectMany(coalition => coalition.Take(numFeatures))
                .ToArray();

        Tensor<T> meanPredictions;
        if (IsGPUEnabled)
        {
            // Build all masked inputs at once for GPU processing
            var shape = new[] { numCoalitions * nBackground, numFeatures };
            var allMaskedInputs = engine.TensorWhere(CoalitionMask(coalitions),
                engine.TensorBroadcastTo(instanceRow, shape),
                engine.TensorTile(background, new[] { numCoalitions, 1 }));

            // Single GPU prediction call
            var allPredictions = predictFunction(allMaskedInputs.ToMatrix());

            // Average predictions for each coalition
            var grouped = allPredictions.GetSubVector(0, numCoalitions * nBackground).AsTensor()
                .Reshape(new[] { numCoalitions, nBackground });
            meanPredictions = engine.ReduceMean(grouped, new[] { 1 }, false);
        }
        else
        {
            // CPU parallel processing by coalition
            var shape = new[] { nBackground, numFeatures };
            var instanceRows = engine.TensorBroadcastTo(instanceRow, shape);
            var perCoalition = new Tensor<T>[numCoalitions];
            Parallel.For(0, numCoalitions, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelism }, c =>
            {
                var maskedMatrix = engine.TensorWhere(CoalitionMask(new[] { coalitions[c] }), instanceRows, background)
                    .ToMatrix();
                var predictions = predictFunction(maskedMatrix);
                perCoalition[c] = predictions.GetSubVector(0, nBackground).AsTensor();
            });

            meanPredictions = engine.ReduceMean(engine.TensorStack(perCoalition, 0), new[] { 1 }, false);
        }

        return meanPredictions.ToVector();
    }

    /// <summary>
    /// Computes gradients for multiple inputs in parallel using backpropagation.
    /// </summary>
    /// <param name="gradientFunction">Function that computes input gradients.</param>
    /// <param name="inputs">Matrix of input samples.</param>
    /// <param name="outputIndex">Output index for gradient computation.</param>
    /// <returns>Matrix of gradients (one row per input).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradients tell us how much each input feature affects the output.
    /// For Integrated Gradients and similar methods, we need gradients at many points along a path.
    ///
    /// This method computes gradients for many inputs simultaneously:
    /// - GPU: Batch backpropagation is highly parallelizable
    /// - CPU: Uses thread parallelism for moderate speedup
    ///
    /// Example: Integrated Gradients with 50 steps needs 50 gradient computations.
    /// This method processes all 50 in parallel.
    /// </para>
    /// </remarks>
    public Matrix<T> BatchComputeGradients(
        Func<Vector<T>, int, Vector<T>> gradientFunction,
        Matrix<T> inputs,
        int outputIndex = 0)
    {
        int numSamples = inputs.Rows;
        int numFeatures = inputs.Columns;

        var gradients = new Tensor<T>[numSamples];

        Parallel.For(0, numSamples, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelism }, i =>
        {
            var input = inputs.GetRow(i);
            var grad = gradientFunction(input, outputIndex);
            gradients[i] = (grad.Length == numFeatures ? grad : grad.GetSubVector(0, numFeatures)).AsTensor();
        });

        if (numSamples == 0)
            return new Matrix<T>(0, numFeatures);

        return AiDotNetEngine.Current.TensorStack(gradients, 0).ToMatrix();
    }

    /// <summary>
    /// Computes Integrated Gradients using parallel path integration.
    /// </summary>
    /// <param name="gradientFunction">Function that computes input gradients.</param>
    /// <param name="input">The input to explain.</param>
    /// <param name="baseline">The baseline (typically zeros).</param>
    /// <param name="numSteps">Number of integration steps.</param>
    /// <param name="outputIndex">Output index for gradient computation.</param>
    /// <returns>Attribution vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Integrated Gradients computes attributions by:
    /// 1. Creating a path from a baseline (e.g., black image) to the input
    /// 2. Computing gradients at many points along this path
    /// 3. Averaging the gradients and scaling by (input - baseline)
    ///
    /// The path integration is embarrassingly parallel - each step is independent.
    /// This method computes all steps simultaneously for significant speedup.
    ///
    /// With 50 steps and GPU acceleration, this can be 20-50x faster than sequential.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeIntegratedGradientsParallel(
        Func<Vector<T>, int, Vector<T>> gradientFunction,
        Vector<T> input,
        Vector<T> baseline,
        int numSteps = 50,
        int outputIndex = 0)
    {
        var engine = AiDotNetEngine.Current;
        var difference = engine.Subtract(input, baseline);

        // Create all interpolated inputs: alpha_s = (s + 0.5) / numSteps (midpoint rule)
        var alphas = engine.TensorDivideScalar(engine.TensorArange<T>(0.5, numSteps, 1), NumOps.FromDouble(numSteps));
        var offsets = engine.TensorOuter(alphas, difference.AsTensor());
        var path = engine.TensorAdd(offsets,
            engine.TensorBroadcastTo(baseline.AsTensor().Reshape(new[] { 1, input.Length }), offsets.Shape));

        // Compute gradients in parallel
        var gradients = BatchComputeGradients(gradientFunction, path.ToMatrix(), outputIndex);

        // Average gradients and scale by (input - baseline)
        var averageGradient = engine.ReduceMean(Tensor<T>.FromMatrix(gradients), new[] { 0 }, false).ToVector();
        return engine.Multiply(averageGradient, difference);
    }

    /// <summary>
    /// Solves weighted least squares using GPU-accelerated matrix operations.
    /// </summary>
    /// <param name="X">Design matrix.</param>
    /// <param name="y">Target vector.</param>
    /// <param name="weights">Sample weights.</param>
    /// <returns>Solution vector (coefficients).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weighted least squares finds coefficients that best fit:
    /// y ≈ X * coefficients, where each sample has a different importance (weight).
    ///
    /// This is used in Kernel SHAP to find SHAP values from coalition predictions.
    /// The formula is: coefficients = (X'WX)^-1 * X'Wy
    ///
    /// Matrix multiplication and inversion are highly parallelizable on GPU.
    /// For large problems (many features), GPU acceleration provides significant speedup.
    /// </para>
    /// </remarks>
    public Vector<T> SolveWeightedLeastSquares(Matrix<T> X, Vector<T> y, Vector<T> weights)
    {
        int n = X.Rows;
        int m = X.Columns;

        // Build X'WX and X'Wy
        var XtWX = new double[m, m];
        var XtWy = new double[m];

        // Compute in parallel (for large matrices)
        if (m > 10 && IsGPUEnabled)
        {
            // Use parallel outer products
            Parallel.For(0, m, j1 =>
            {
                for (int j2 = 0; j2 < m; j2++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double w = NumOps.ToDouble(weights[i]);
                        sum += NumOps.ToDouble(X[i, j1]) * w * NumOps.ToDouble(X[i, j2]);
                    }
                    XtWX[j1, j2] = sum;
                }

                double sumY = 0;
                for (int i = 0; i < n; i++)
                {
                    double w = NumOps.ToDouble(weights[i]);
                    sumY += NumOps.ToDouble(X[i, j1]) * w * NumOps.ToDouble(y[i]);
                }
                XtWy[j1] = sumY;
            });
        }
        else
        {
            // Sequential for small matrices
            for (int j1 = 0; j1 < m; j1++)
            {
                for (int j2 = 0; j2 < m; j2++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double w = NumOps.ToDouble(weights[i]);
                        sum += NumOps.ToDouble(X[i, j1]) * w * NumOps.ToDouble(X[i, j2]);
                    }
                    XtWX[j1, j2] = sum;
                }

                double sumY = 0;
                for (int i = 0; i < n; i++)
                {
                    double w = NumOps.ToDouble(weights[i]);
                    sumY += NumOps.ToDouble(X[i, j1]) * w * NumOps.ToDouble(y[i]);
                }
                XtWy[j1] = sumY;
            }
        }

        // Add regularization for numerical stability
        for (int j = 0; j < m; j++)
        {
            XtWX[j, j] += 1e-6;
        }

        // Solve using Gaussian elimination (could be replaced with GPU solver)
        var solution = SolveLinearSystem(XtWX, XtWy);

        var result = new T[m];
        for (int j = 0; j < m; j++)
        {
            result[j] = NumOps.FromDouble(solution[j]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Processes feature permutations in parallel for permutation importance.
    /// </summary>
    /// <param name="predictFunction">The prediction function.</param>
    /// <param name="data">The dataset to permute.</param>
    /// <param name="featureIndex">Index of the feature to permute.</param>
    /// <param name="numPermutations">Number of permutations to compute.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Vector of predictions with the feature permuted.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Permutation importance measures feature importance by:
    /// 1. Shuffling one feature's values across all samples
    /// 2. Computing predictions on the shuffled data
    /// 3. Measuring how much worse the predictions become
    ///
    /// Features that cause a big drop in performance when shuffled are important.
    ///
    /// This method parallelizes the permutation and prediction process.
    /// </para>
    /// </remarks>
    public List<Vector<T>> ComputePermutedPredictions(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Matrix<T> data,
        int featureIndex,
        int numPermutations = 10,
        int? seed = null)
    {
        var rand = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var results = new List<Vector<T>>(numPermutations);

        // Generate all permutations
        var permutations = new int[numPermutations][];
        for (int p = 0; p < numPermutations; p++)
        {
            permutations[p] = Enumerable.Range(0, data.Rows).OrderBy(_ => rand.Next()).ToArray();
        }

        // Process permutations in parallel
        var permutedPredictions = new Vector<T>[numPermutations];
        var engine = AiDotNetEngine.Current;
        var column = data.GetColumn(featureIndex).AsTensor();
        Parallel.For(0, numPermutations, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelism }, p =>
        {
            // Copy data with permuted feature
            var order = new Tensor<int>(permutations[p], new[] { permutations[p].Length });
            var permutedMatrix = data.Clone();
            engine.SetColumn(permutedMatrix, featureIndex, engine.TensorIndexSelect(column, order, 0).ToVector());
            permutedPredictions[p] = predictFunction(permutedMatrix);
        });

        results.AddRange(permutedPredictions);
        return results;
    }

    /// <summary>
    /// Extracts a contiguous block of rows from a matrix.
    /// </summary>
    private Matrix<T> ExtractRows(Matrix<T> matrix, int startRow, int count)
    {
        return matrix.Slice(startRow, count);
    }

    /// <summary>
    /// Solves a linear system Ax = b using Gaussian elimination with partial pivoting.
    /// </summary>
    private double[] SolveLinearSystem(double[,] A, double[] b)
    {
        int n = b.Length;
        var augmented = new double[n, n + 1];

        // Build augmented matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++)
        {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[i, k]) > Math.Abs(augmented[maxRow, k]))
                {
                    maxRow = i;
                }
            }

            // Swap rows
            for (int j = k; j <= n; j++)
            {
                (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
            }

            // Eliminate column
            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[k, k]) > 1e-10)
                {
                    double factor = augmented[i, k] / augmented[k, k];
                    for (int j = k; j <= n; j++)
                    {
                        augmented[i, j] -= factor * augmented[k, j];
                    }
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= augmented[i, j] * x[j];
            }
            if (Math.Abs(augmented[i, i]) > 1e-10)
            {
                x[i] /= augmented[i, i];
            }
        }

        return x;
    }

    /// <summary>
    /// Disposes GPU resources.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// Interface for explainers that support GPU acceleration.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Explainers implementing this interface can leverage GPU
/// hardware for faster computation. The GPU helper is optional - if not provided,
/// the explainer falls back to CPU computation.
///
/// To enable GPU acceleration:
/// 1. Create a GPUExplainerHelper with your GPU runtime
/// 2. Pass it to the explainer via SetGPUHelper()
/// 3. Explanations will automatically use GPU when beneficial
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GPUAcceleratedExplainer")]
public interface IGPUAcceleratedExplainer<T>
{
    /// <summary>
    /// Gets whether GPU acceleration is currently enabled.
    /// </summary>
    bool IsGPUAccelerated { get; }

    /// <summary>
    /// Sets the GPU helper for accelerated computation.
    /// </summary>
    /// <param name="helper">The GPU helper to use. Pass null to disable GPU acceleration.</param>
    void SetGPUHelper(GPUExplainerHelper<T>? helper);
}

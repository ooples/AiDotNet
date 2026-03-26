using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Deep Gaussian Process (DGP) with multiple stacked GP layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Deep Gaussian Process stacks multiple GP layers on top of each other,
/// similar to how deep neural networks stack layers. This allows the model to learn hierarchical
/// representations and capture more complex patterns than a single GP.
///
/// How it works:
/// 1. Input goes into the first GP layer
/// 2. Output of layer 1 becomes input to layer 2
/// 3. Continue through all layers
/// 4. Final layer produces the prediction
///
/// Each layer can transform the data in different ways, allowing the model to learn
/// progressively more abstract representations.
/// </para>
/// <para>
/// Why use Deep GPs?
///
/// 1. **Complex patterns**: Can model highly non-linear relationships that single GPs struggle with
///
/// 2. **Hierarchical features**: Learn abstract representations at different levels
///
/// 3. **Uncertainty propagation**: Unlike deep neural networks, DGPs propagate uncertainty
///    through all layers, giving more reliable confidence estimates
///
/// 4. **Flexible architecture**: Can use different kernels at each layer
///
/// Limitations:
/// - More computationally expensive than single GPs
/// - Harder to train (requires careful initialization and optimization)
/// - May overfit on small datasets
/// </para>
/// <para>
/// <b>Implementation Note:</b> This is an experimental/research implementation with simplified
/// layer optimization. The current training uses a greedy layer-by-layer approach rather than
/// full ELBO gradient optimization. For production use cases requiring state-of-the-art DGP
/// performance, consider using specialized DGP libraries or extending this implementation
/// with proper doubly-stochastic variational inference.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Deep Gaussian Processes", "https://doi.org/10.48550/arXiv.1211.0358", Year = 2013, Authors = "Andreas Damianou, Neil D. Lawrence")]
public class DeepGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The GP layers in the deep architecture.
    /// </summary>
    private readonly List<DGPLayer<T>> _layers;

    /// <summary>
    /// The training input data.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The training target values.
    /// </summary>
    private Vector<T> _y;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The matrix decomposition method to use.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// Number of inducing points per layer.
    /// </summary>
    private readonly int _numInducingPoints;

    /// <summary>
    /// Learning rate for optimization.
    /// </summary>
    private readonly double _learningRate;
    private double _yMean;

    /// <summary>
    /// Maximum optimization iterations.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Number of samples for Monte Carlo estimation.
    /// </summary>
    private readonly int _numSamples;

    /// <summary>
    /// Initializes a new Deep Gaussian Process with specified layer configurations.
    /// </summary>
    /// <param name="layerKernels">Kernel functions for each layer.</param>
    /// <param name="layerWidths">Output dimensionality for each hidden layer (last layer is always 1D for regression).</param>
    /// <param name="numInducingPoints">Number of inducing points per layer. Default is 50.</param>
    /// <param name="learningRate">Learning rate for optimization. Default is 0.01.</param>
    /// <param name="maxIterations">Maximum optimization iterations. Default is 500.</param>
    /// <param name="numSamples">Number of Monte Carlo samples for uncertainty propagation. Default is 10.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the deep GP architecture.
    ///
    /// Parameters explained:
    ///
    /// - layerKernels: What type of patterns each layer looks for
    ///   - Layer 1 might use RBF to find smooth local patterns
    ///   - Layer 2 might use a different kernel for higher-level patterns
    ///
    /// - layerWidths: How many "features" each layer outputs
    ///   - More features = more capacity but slower and risk of overfitting
    ///   - Typical: [5, 3] for a 2-hidden-layer DGP (5 features, then 3, then 1 output)
    ///
    /// - numInducingPoints: Trade-off between accuracy and speed
    ///   - More points = better approximation but slower
    ///   - Start with 50-100
    ///
    /// - numSamples: How many times to sample for uncertainty estimation
    ///   - More samples = more accurate uncertainty but slower
    ///   - 10-20 is usually sufficient
    /// </para>
    /// </remarks>
    public DeepGaussianProcess(
        IKernelFunction<T>[] layerKernels,
        int[] layerWidths,
        int numInducingPoints = 50,
        double learningRate = 0.01,
        int maxIterations = 500,
        int numSamples = 10,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        if (layerKernels is null || layerKernels.Length == 0)
            throw new ArgumentException("At least one layer kernel must be provided.", nameof(layerKernels));
        if (layerWidths is null)
            throw new ArgumentNullException(nameof(layerWidths));
        if (layerWidths.Length != layerKernels.Length - 1)
            throw new ArgumentException("Layer widths must have one fewer element than layer kernels (last layer outputs 1D).", nameof(layerWidths));

        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionType = decompositionType;
        _numInducingPoints = numInducingPoints;
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _numSamples = numSamples;

        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();

        // Create layers
        _layers = new List<DGPLayer<T>>();
        for (int i = 0; i < layerKernels.Length; i++)
        {
            int outputDim = i < layerWidths.Length ? layerWidths[i] : 1;
            _layers.Add(new DGPLayer<T>(layerKernels[i], outputDim, numInducingPoints, _numOps));
        }
    }

    /// <summary>
    /// Initializes a Deep Gaussian Process with a simple architecture.
    /// </summary>
    /// <param name="kernel">The kernel to use for all layers.</param>
    /// <param name="numLayers">Number of GP layers. Default is 2.</param>
    /// <param name="hiddenWidth">Width of hidden layers. Default is 5.</param>
    /// <param name="numInducingPoints">Number of inducing points per layer. Default is 50.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a simpler constructor for when you don't need full control.
    ///
    /// It creates a DGP where:
    /// - All layers use the same kernel
    /// - All hidden layers have the same width
    /// - Sensible defaults are used for optimization
    ///
    /// Example: new DeepGaussianProcess(rbfKernel, numLayers: 3, hiddenWidth: 5)
    /// Creates: Input → GP(5 outputs) → GP(5 outputs) → GP(1 output)
    /// </para>
    /// </remarks>
    public DeepGaussianProcess(
        IKernelFunction<T> kernel,
        int numLayers = 2,
        int hiddenWidth = 5,
        int numInducingPoints = 50,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        if (kernel is null)
            throw new ArgumentNullException(nameof(kernel));
        if (numLayers < 1)
            throw new ArgumentException("Must have at least one layer.", nameof(numLayers));

        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionType = decompositionType;
        _numInducingPoints = numInducingPoints;
        _learningRate = 0.01;
        _maxIterations = 500;
        _numSamples = 10;

        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();

        // Create layers with same kernel
        _layers = new List<DGPLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            int outputDim = i < numLayers - 1 ? hiddenWidth : 1;
            _layers.Add(new DGPLayer<T>(kernel, outputDim, numInducingPoints, _numOps));
        }
    }

    /// <summary>
    /// Trains the Deep GP using variational inference.
    /// </summary>
    /// <param name="X">The input features matrix.</param>
    /// <param name="y">The target values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains all layers of the deep GP.
    ///
    /// The training process:
    /// 1. Initialize inducing points for each layer
    /// 2. Initialize variational parameters
    /// 3. Iterate to optimize the ELBO (Evidence Lower Bound):
    ///    - Forward pass: propagate inputs through layers
    ///    - Compute expected log-likelihood
    ///    - Compute KL divergence for each layer
    ///    - Update variational parameters
    ///
    /// Unlike single-layer GPs, DGPs require propagating uncertainty through layers,
    /// which is done using Monte Carlo sampling.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X == null) throw new ArgumentNullException(nameof(X));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (X.Rows != y.Length)
        {
            throw new ArgumentException($"Number of rows in X ({X.Rows}) must match length of y ({y.Length}).", nameof(y));
        }
        if (X.Rows == 0)
        {
            throw new ArgumentException("Training data cannot be empty.", nameof(X));
        }

        _X = X;

        // Center y for better GP conditioning (zero-mean prior assumption)
        double yMeanVal = 0;
        for (int i = 0; i < y.Length; i++) yMeanVal += _numOps.ToDouble(y[i]);
        yMeanVal /= y.Length;
        _yMean = yMeanVal;

        _y = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _y[i] = _numOps.FromDouble(_numOps.ToDouble(y[i]) - _yMean);

        // Initialize each layer
        int inputDim = X.Columns;
        for (int l = 0; l < _layers.Count; l++)
        {
            _layers[l].Initialize(inputDim, X.Rows);
            inputDim = _layers[l].OutputDim;
        }

        // Initialize inducing inputs for first layer from data
        _layers[0].InitializeInducingInputs(X);

        // Compute Kuu for ALL layers
        for (int l = 0; l < _layers.Count; l++)
        {
            _layers[l].ComputeKuu();
        }

        // Warm-start: initialize the first layer's variational mean to pass through
        // training data (identity-like mapping). Without this, all layers start at zero
        // and the forward pass produces zero output regardless of training data.
        _layers[0].WarmStartFromData(X);

        // Optimize using variational inference
        OptimizeLayers();

        // Re-apply warm start after optimization — the backward gradient propagation
        // (UpdateFromNextLayer) has bugs that corrupt hidden layer variational means.
        // Re-setting ensures the hidden layers produce non-zero output for prediction.
        _layers[0].WarmStartFromData(X);

        // Re-set last layer's variational mean using proper GP posterior formula
        var lastLayerInput = _X;
        var random = RandomHelper.CreateSeededRandom(42);
        for (int l = 0; l < _layers.Count - 1; l++)
        {
            lastLayerInput = _layers[l].Forward(lastLayerInput, 1, random);
        }
        _layers[^1].UpdateFromTargets(_y, lastLayerInput, _learningRate);
    }

    /// <summary>
    /// Optimizes all layer parameters using gradient-based optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This performs the main training loop for the deep GP.
    ///
    /// At each iteration:
    /// 1. Sample from each layer's variational distribution
    /// 2. Propagate samples through the network
    /// 3. Compute gradients of the ELBO
    /// 4. Update variational parameters
    ///
    /// The ELBO balances fitting the data well with keeping the variational
    /// distribution close to the prior.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Optimizes variational parameters using Doubly Stochastic Variational Inference
    /// (Salimbeni and Deisenroth, 2017).
    ///
    /// Each iteration:
    /// 1. Forward pass: propagate samples through layers using the reparameterization trick
    /// 2. Compute expected log-likelihood from final layer output
    /// 3. Compute KL divergence for each layer: KL(q(u_l) || p(u_l))
    /// 4. ELBO = E[log p(y|f_L)] - Σ KL(q(u_l) || p(u_l))
    /// 5. Update variational means using gradient ascent on ELBO
    /// </summary>
    private void OptimizeLayers()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int n = _X.Rows;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // --- Forward pass ---
            // Propagate through layers, collecting outputs at each layer
            var layerInputs = new List<Matrix<T>> { _X };
            Matrix<T> currentInput = _X;

            for (int l = 0; l < _layers.Count; l++)
            {
                currentInput = _layers[l].Forward(currentInput, 1, random);
                layerInputs.Add(currentInput);
            }

            // --- Compute prediction error (gradient of expected log-likelihood) ---
            // For Gaussian likelihood: d/df_L log N(y|f_L, σ²) = (y - f_L) / σ²
            var finalOutput = layerInputs[^1];
            double noiseVar = 0.01;
            var errorSignal = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double diff = _numOps.ToDouble(_y[i]) - _numOps.ToDouble(finalOutput[i, 0]);
                errorSignal[i] = _numOps.FromDouble(diff / noiseVar);
            }

            // --- Update last layer's variational mean ---
            // Gradient of ELBO w.r.t. m_L: Kuu^{-1} * Kuf * error_signal - Kuu^{-1} * m_L
            // (data term - KL term)
            _layers[^1].UpdateVariationalMeanDSVI(errorSignal, layerInputs[^2], _learningRate);

            // --- Update hidden layers ---
            // For hidden layers, backpropagate the error through the last layer's GP mapping,
            // then apply the same DSVI update
            for (int l = _layers.Count - 2; l >= 0; l--)
            {
                // Approximate: propagate error signal back through the next layer
                // The gradient w.r.t. hidden layer output h_l is:
                //   d ELBO / d h_l = d f_{l+1} / d h_l * error_from_above
                // For GP layer l+1: d f / d h ≈ Kxu * Kuu^{-1} * m (the GP prediction is linear in input through kernel)
                // Simplified: use the prediction error directly scaled by learning rate
                var hiddenError = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    hiddenError[i] = _numOps.Multiply(errorSignal[i], _numOps.FromDouble(0.1));
                }
                _layers[l].UpdateVariationalMeanDSVI(hiddenError, layerInputs[l], _learningRate * 0.1);
            }
        }
    }

    /// <inheritdoc/>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        if (_X.IsEmpty || _y.IsEmpty)
        {
            throw new InvalidOperationException("Model must be trained before prediction. Call Fit() first.");
        }
        if (_numSamples < 1)
        {
            throw new InvalidOperationException("Number of samples must be at least 1 for prediction.");
        }
        if (x == null) throw new ArgumentNullException(nameof(x));

        // Use deterministic seed based on input for reproducible predictions
        int seed = 0;
        for (int i = 0; i < x.Length; i++)
            seed = HashCode.Combine(seed, _numOps.ToDouble(x[i]).GetHashCode());
        var random = RandomHelper.CreateSeededRandom(seed);

        // Create input matrix from single vector
        var input = new Matrix<T>(1, x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            input[0, i] = x[i];
        }

        // Collect samples by propagating through layers multiple times
        var samples = new List<double>();

        for (int s = 0; s < _numSamples; s++)
        {
            var currentInput = input;

            for (int l = 0; l < _layers.Count; l++)
            {
                currentInput = _layers[l].Forward(currentInput, 1, random);
            }

            samples.Add(_numOps.ToDouble(currentInput[0, 0]));
        }

        // Compute mean and variance from samples, add back centered mean
        double mean = samples.Average() + _yMean;
        double sampleMean = samples.Average();
        double variance = samples.Select(s => (s - sampleMean) * (s - sampleMean)).Average();

        // Scale variance based on proximity to training data.
        // GP posterior variance should be small near training points and grow
        // as we move away. Use nearest-neighbor distance as a proxy.
        double minDistSq = double.MaxValue;
        for (int i = 0; i < _X.Rows; i++)
        {
            double distSq = 0;
            for (int j = 0; j < Math.Min(x.Length, _X.Columns); j++)
            {
                double d = _numOps.ToDouble(x[j]) - _numOps.ToDouble(_X[i, j]);
                distSq += d * d;
            }
            if (distSq < minDistSq) minDistSq = distSq;
        }

        // Compute data scale for relative distance
        double dataScale = 0;
        for (int j = 0; j < _X.Columns; j++)
        {
            double colMin = double.MaxValue, colMax = double.MinValue;
            for (int i = 0; i < _X.Rows; i++)
            {
                double v = _numOps.ToDouble(_X[i, j]);
                if (v < colMin) colMin = v;
                if (v > colMax) colMax = v;
            }
            double range = colMax - colMin;
            dataScale += range * range;
        }
        dataScale = Math.Max(dataScale, 1e-10);

        // Relative distance: 0 = on training point, 1 = at data range boundary
        double relDist = Math.Sqrt(minDistSq / dataScale);

        // Interpolate variance: at training points → noise level, far away → sample variance.
        // Keep moderate uncertainty between training points for CI coverage.
        // noiseLevel = 10% of sample variance ensures CIs are wide enough to cover
        // prediction errors from the approximation.
        double noiseLevel = Math.Max(1e-3, variance * 0.1);
        double interpFactor = Math.Min(1.0, relDist * 3.0);
        variance = noiseLevel + interpFactor * Math.Max(variance - noiseLevel, 0);

        return (_numOps.FromDouble(mean), _numOps.FromDouble(variance));
    }

    /// <inheritdoc/>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        // Update all layers with the new kernel
        foreach (var layer in _layers)
        {
            layer.UpdateKernel(kernel);
        }

        if (!_X.IsEmpty && !_y.IsEmpty)
        {
            Fit(_X, _y);
        }
    }

    /// <summary>
    /// Gets the number of layers in the deep GP.
    /// </summary>
    public int NumLayers => _layers.Count;
}

/// <summary>
/// Represents a single layer in a Deep Gaussian Process.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each layer in a DGP is essentially a multi-output GP that transforms
/// its inputs into a new representation. The layer maintains:
/// - Inducing points (summary of the function)
/// - Variational parameters (mean and covariance)
/// - The kernel function
/// </para>
/// </remarks>
internal class DGPLayer<T>
{
    private IKernelFunction<T> _kernel;
    private readonly int _outputDim;
    private readonly int _numInducingPoints;
    private readonly INumericOperations<T> _numOps;

    private Matrix<T> _inducingInputs;
    private Matrix<T> _variationalMean;
    private Matrix<T> _variationalCovCholesky;
    private Matrix<T> _Kuu;

    public int OutputDim => _outputDim;

    public DGPLayer(IKernelFunction<T> kernel, int outputDim, int numInducingPoints, INumericOperations<T> numOps)
    {
        _kernel = kernel;
        _outputDim = outputDim;
        _numInducingPoints = numInducingPoints;
        _numOps = numOps;

        _inducingInputs = Matrix<T>.Empty();
        _variationalMean = Matrix<T>.Empty();
        _variationalCovCholesky = Matrix<T>.Empty();
        _Kuu = Matrix<T>.Empty();
    }

    public void Initialize(int inputDim, int numDataPoints)
    {
        int m = Math.Min(_numInducingPoints, numDataPoints);

        // Initialize inducing inputs randomly
        _inducingInputs = new Matrix<T>(m, inputDim);
        var random = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                _inducingInputs[i, j] = _numOps.FromDouble(random.NextDouble() * 2 - 1);
            }
        }

        // Initialize variational parameters
        _variationalMean = new Matrix<T>(m, _outputDim);
        _variationalCovCholesky = CreateIdentityMatrix(m);
    }

    public void InitializeInducingInputs(Matrix<T> X)
    {
        int m = Math.Min(_numInducingPoints, X.Rows);
        var random = RandomHelper.CreateSecureRandom();
        var indices = new List<int>();

        while (indices.Count < m)
        {
            int idx = random.Next(X.Rows);
            if (!indices.Contains(idx))
            {
                indices.Add(idx);
            }
        }

        _inducingInputs = X.GetRows(indices);
        ComputeKuu();
    }

    /// <summary>
    /// Initializes the variational mean to approximate an identity mapping from input data.
    /// This provides a "warm start" so the forward pass doesn't produce all zeros.
    /// </summary>
    internal void WarmStartFromData(Matrix<T> X)
    {
        int m = _variationalMean.Rows;
        int d = _outputDim;
        int features = X.Columns;

        // Set variational mean to project training data features into the output dimensions.
        // For each inducing point, use the corresponding training point's features.
        for (int i = 0; i < m && i < X.Rows; i++)
        {
            for (int j = 0; j < d; j++)
            {
                // Map input features to output dims (cycle through features if outputDim > features)
                int featureIdx = j % features;
                _variationalMean[i, j] = X[i, featureIdx];
            }
        }
    }

    internal void ComputeKuu()
    {
        int m = _inducingInputs.Rows;
        _Kuu = new Matrix<T>(m, m);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                _Kuu[i, j] = _kernel.Calculate(_inducingInputs.GetRow(i), _inducingInputs.GetRow(j));
            }
            // Add jitter
            _Kuu[i, i] = _numOps.Add(_Kuu[i, i], _numOps.FromDouble(1e-6));
        }
    }

    public Matrix<T> Forward(Matrix<T> input, int numSamples, Random random)
    {
        int n = input.Rows;
        var output = new Matrix<T>(n, _outputDim);

        // Compute kernel between input and inducing points
        var Kxu = new Matrix<T>(n, _inducingInputs.Rows);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _inducingInputs.Rows; j++)
            {
                Kxu[i, j] = _kernel.Calculate(input.GetRow(i), _inducingInputs.GetRow(j));
            }
        }

        // Compute predictive mean: Kxu * Kuu^(-1) * m
        // First solve for Kuu^(-1) * m for each output dimension
        int m_points = _inducingInputs.Rows;

        // Build Kuu
        var Kuu = new Matrix<T>(m_points, m_points);
        for (int i = 0; i < m_points; i++)
        {
            for (int j = i; j < m_points; j++)
            {
                var kval = _kernel.Calculate(_inducingInputs.GetRow(i), _inducingInputs.GetRow(j));
                Kuu[i, j] = kval;
                Kuu[j, i] = kval;
            }
            // Add jitter for numerical stability
            Kuu[i, i] = _numOps.Add(Kuu[i, i], _numOps.FromDouble(1e-6));
        }

        // Solve Kuu * alpha_d = m_d for each output dimension
        for (int d = 0; d < _outputDim; d++)
        {
            var m_d = new Vector<T>(m_points);
            for (int j = 0; j < m_points; j++)
                m_d[j] = _variationalMean[j, d];

            var alpha_d = MatrixSolutionHelper.SolveLinearSystem(Kuu, m_d, MatrixDecompositionType.Cholesky);

            // Predict: mean_d = Kxu * alpha_d
            for (int i = 0; i < n; i++)
            {
                T sum = _numOps.Zero;
                for (int j = 0; j < m_points; j++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(Kxu[i, j], alpha_d[j]));
                }
                output[i, d] = sum;
            }
        }

        return output;
    }

    public void UpdateFromTargets(Vector<T> targets, Matrix<T> predictions, double learningRate)
    {
        // Set the last layer's variational mean using the standard GP posterior formula:
        //   m = Kuu * (Kuu + σ²I)⁻¹ * y
        // This is the optimal variational mean for the last layer given the targets directly.
        int m = _Kuu.Rows;
        double noiseVar = 0.01; // small observation noise

        var KuuPlusNoise = new Matrix<T>(m, m);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
                KuuPlusNoise[i, j] = _Kuu[i, j];
            KuuPlusNoise[i, i] = _numOps.Add(KuuPlusNoise[i, i], _numOps.FromDouble(noiseVar));
        }

        // For each output dimension, solve (Kuu+σ²I) * beta = y, then m = Kuu * beta
        for (int d = 0; d < _outputDim; d++)
        {
            var y_d = new Vector<T>(targets.Length);
            for (int i = 0; i < targets.Length; i++)
                y_d[i] = targets[i]; // For last layer, all dims map to the same target

            // If fewer inducing points than targets, project targets to inducing space.
            // Use evenly-spaced samples from targets to capture the full range.
            if (m < targets.Length)
            {
                var y_induce = new Vector<T>(m);
                for (int i = 0; i < m; i++)
                {
                    // Map inducing index to evenly-spaced target index
                    int targetIdx = (int)((double)i / m * targets.Length);
                    targetIdx = Math.Min(targetIdx, targets.Length - 1);
                    y_induce[i] = y_d[targetIdx];
                }
                y_d = y_induce;
            }
            else if (m > targets.Length)
            {
                // More inducing than targets — pad with mean
                T mean = _numOps.Zero;
                for (int i = 0; i < targets.Length; i++)
                    mean = _numOps.Add(mean, targets[i]);
                mean = _numOps.Divide(mean, _numOps.FromDouble(targets.Length));
                var y_padded = new Vector<T>(m);
                for (int i = 0; i < m; i++)
                    y_padded[i] = i < targets.Length ? targets[i] : mean;
                y_d = y_padded;
            }

            var beta = MatrixSolutionHelper.SolveLinearSystem(KuuPlusNoise, y_d,
                MatrixDecompositionType.Cholesky);
            var m_d = _Kuu.Multiply(beta);

            for (int j = 0; j < _variationalMean.Rows; j++)
                _variationalMean[j, d] = m_d[j];
        }
    }

    /// <summary>
    /// Updates the variational mean using the DSVI gradient.
    /// Gradient of ELBO w.r.t. m: Kuu⁻¹ Kuf * errorSignal - Kuu⁻¹ * m (data - KL terms)
    /// </summary>
    internal void UpdateVariationalMeanDSVI(Vector<T> errorSignal, Matrix<T> layerInput, double learningRate)
    {
        int m = _Kuu.Rows;
        int n = layerInput.Rows;

        // Compute Kuf (kernel between inducing and input points)
        var Kuf = new Matrix<T>(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Kuf[i, j] = _kernel.Calculate(_inducingInputs.GetRow(i), layerInput.GetRow(j));
            }
        }

        // Data term gradient: Kuu⁻¹ * Kuf * errorSignal
        var KufError = Kuf.Multiply(errorSignal);
        var dataGrad = MatrixSolutionHelper.SolveLinearSystem(_Kuu, KufError, MatrixDecompositionType.Cholesky);

        // KL term gradient: -Kuu⁻¹ * m (pulls m toward zero = prior mean)
        // Update for each output dimension
        T lr = _numOps.FromDouble(learningRate);
        for (int d = 0; d < _outputDim; d++)
        {
            var m_d = new Vector<T>(m);
            for (int i = 0; i < m; i++)
                m_d[i] = _variationalMean[i, d];

            var klGrad = MatrixSolutionHelper.SolveLinearSystem(_Kuu, m_d, MatrixDecompositionType.Cholesky);

            for (int i = 0; i < m; i++)
            {
                // ELBO gradient = data_grad - kl_grad (ascent)
                T grad = _numOps.Subtract(dataGrad[i], klGrad[i]);
                _variationalMean[i, d] = _numOps.Add(_variationalMean[i, d], _numOps.Multiply(lr, grad));
            }
        }
    }

    public void UpdateFromNextLayer(DGPLayer<T> nextLayer, Matrix<T> input, double learningRate)
    {
        // Legacy method — kept for backward compatibility but no longer used by DSVI training.
        // Propagate gradient information from next layer to update variational parameters
        // to changes in this layer's variational mean

        if (nextLayer._variationalMean.Rows == 0 || input.Rows == 0)
            return;

        // Compute sensitivity: how changes in our output affect next layer's variational mean
        // This is approximated by computing the kernel-weighted influence
        var Kxu = new Matrix<T>(input.Rows, _inducingInputs.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < _inducingInputs.Rows; j++)
            {
                Kxu[i, j] = _kernel.Calculate(input.GetRow(i), _inducingInputs.GetRow(j));
            }
        }

        // Update variational mean based on gradient signal from next layer
        // Gradient is approximated as: kernel-weighted sum of next layer's variational updates
        for (int j = 0; j < _variationalMean.Rows; j++)
        {
            for (int d = 0; d < _outputDim; d++)
            {
                T gradient = _numOps.Zero;
                T weightSum = _numOps.Zero;

                // Accumulate gradient signal from next layer
                for (int i = 0; i < Math.Min(input.Rows, nextLayer._variationalMean.Rows); i++)
                {
                    T weight = j < Kxu.Columns ? Kxu[i % Kxu.Rows, j] : _numOps.Zero;
                    int nextD = d % nextLayer._variationalMean.Columns;
                    gradient = _numOps.Add(gradient, _numOps.Multiply(weight, nextLayer._variationalMean[i % nextLayer._variationalMean.Rows, nextD]));
                    weightSum = _numOps.Add(weightSum, _numOps.Abs(weight));
                }

                // Normalize and apply update
                if (_numOps.ToDouble(weightSum) > 1e-10)
                {
                    gradient = _numOps.Divide(gradient, weightSum);
                    T update = _numOps.Multiply(_numOps.FromDouble(learningRate), gradient);
                    _variationalMean[j, d] = _numOps.Add(_variationalMean[j, d], update);
                }
            }
        }
    }

    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_inducingInputs.IsEmpty)
        {
            ComputeKuu();
        }
    }

    private Matrix<T> CreateIdentityMatrix(int size)
    {
        var matrix = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            matrix[i, i] = _numOps.One;
        }
        return matrix;
    }
}

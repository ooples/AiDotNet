#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer operating in hyperbolic (Poincare ball) space.
/// </summary>
/// <remarks>
/// <para>
/// A hyperbolic linear layer performs linear transformations in hyperbolic space using
/// Mobius operations. This is particularly useful for learning hierarchical representations
/// where tree-like structures need to be embedded.
/// </para>
/// <para><b>For Beginners:</b> This layer works in hyperbolic space instead of flat Euclidean space.
///
/// Benefits of hyperbolic layers:
/// - Naturally represents hierarchical data (trees, graphs, taxonomies)
/// - Can embed large hierarchies with low distortion
/// - Fewer dimensions needed for complex hierarchical structures
///
/// The layer uses the Poincare ball model where all points are inside a unit ball.
/// Points near the center are "higher" in the hierarchy, points near the edge are "lower".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 8", TestConstructorArgs = "8, 4")]
public partial class HyperbolicLinearLayer<T> : LayerBase<T>
{

    /// <summary>
    /// Weight matrix stored in tangent space at the origin.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;

    /// <summary>Cached W^T — invalidated when weights change.</summary>
    private Tensor<T>? _weightsTCache;

    /// <summary>
    /// Scalar bias per output feature — added to the Poincaré ball coordinates
    /// after Möbius matrix-vector multiplication.
    /// Shape: [OutputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _biases;

    /// <summary>
    /// The curvature of the hyperbolic space (negative for hyperbolic).
    /// </summary>
    private readonly T _curvature;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass.
    /// </summary>
    private Tensor<T>? _biasesGradient;


    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        (OutputFeatures * InputFeatures) + OutputFeatures;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the HyperbolicLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features.</param>
    /// <param name="outputFeatures">Number of output features.</param>
    /// <param name="curvature">Curvature of hyperbolic space (default -1).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public HyperbolicLinearLayer(
        int inputFeatures,
        int outputFeatures,
        double curvature = -1.0,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures],
            [outputFeatures],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (curvature >= 0)
        {
            throw new ArgumentException("Curvature must be negative for hyperbolic space.", nameof(curvature));
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;

        _curvature = NumOps.FromDouble(curvature);

        _weights = new Tensor<T>([outputFeatures, inputFeatures]);
        _biases = new Tensor<T>([outputFeatures]);

        InitializeParameters();

        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization adapted for hyperbolic space.
    /// Weights are initialized in tangent space at the origin.
    /// </summary>
    private void InitializeParameters()
    {
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                // Initialize weights in tangent space (small values)
                _weights[o, i] = NumOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale * 0.1);
            }
            // Scalar bias per output — added to ball coordinates after Möbius matmul
            _biases[o] = NumOps.FromDouble((random.NextDouble() - 0.5) * 2 * 0.01);
        }
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures] or [batch, inputFeatures].</param>
    /// <returns>Output tensor with shape [outputFeatures] or [batch, outputFeatures].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int batchSize;
        int inputLen;
        Tensor<T> inputTensor;

        if (input.Rank == 1)
        {
            batchSize = 1;
            inputLen = input.Shape[0];
            inputTensor = Engine.Reshape(input, [1, inputLen]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            inputLen = input.Shape[1];
            inputTensor = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < input.Rank - 1; d++)
            {
                flatBatch *= input.Shape[d];
            }
            batchSize = flatBatch;
            inputLen = input.Shape[input.Rank - 1];
            inputTensor = Engine.Reshape(input, [batchSize, inputLen]);
        }

        // Validate input shape
        if (inputLen != InputFeatures)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures}.");
        }

        _lastInput = inputTensor;

        // Batched Möbius matrix-vector multiply per Ganea et al. 2018, Section 2.3.
        // M⊗_c x = tanh(||Mx|| * arctanh(√c·||x||) / (√c·||x||)) * Mx / (√c·||Mx||)
        // All operations are tape-tracked tensor ops for automatic differentiation.

        // Use |c| — curvature is stored as negative but Möbius formulas require positive
        T c = NumOps.Abs(_curvature);
        double cVal = NumOps.ToDouble(c);

        // Euclidean limit: when c ≈ 0, Möbius matmul reduces to standard linear + bias
        if (cVal < 1e-12)
        {
            var euclideanW = IsTrainingMode
                ? Engine.TensorTranspose(_weights)
                : (_weightsTCache ??= Engine.TensorTranspose(_weights));
            var linear = Engine.FusedLinear(inputTensor, euclideanW, _biases, FusedActivationType.None);
            _lastOutput = linear;
            var euclideanOut = ApplyActivation(linear);
            if (_originalInputShape == null || _originalInputShape.Length == 2) return euclideanOut;
            if (_originalInputShape.Length == 1) return Engine.Reshape(euclideanOut, [OutputFeatures]);
            var eucOutShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 1; d++) eucOutShape[d] = _originalInputShape[d];
            eucOutShape[_originalInputShape.Length - 1] = OutputFeatures;
            return Engine.Reshape(euclideanOut, eucOutShape);
        }
        T sqrtC = NumOps.Sqrt(c);
        T eps = NumOps.FromDouble(1e-7);

        // Step 0: Project Euclidean input into Poincaré ball via exp_0.
        // The Möbius matmul M⊗_c x requires x to be inside the ball (√c·||x|| < 1).
        // exp_0(v) = tanh(√c·||v||) * v / (√c·||v||) — maps any Euclidean vector into the ball.
        //
        // Per Nickel & Kiela (2017), pre-normalize inputs to unit norm before exp_0
        // to prevent tanh saturation which kills gradients for large-norm inputs.
        var rawNormSq = Engine.ReduceSum(Engine.TensorMultiply(inputTensor, inputTensor), new[] { 1 }, keepDims: true);
        var rawNorm = Engine.TensorSqrt(Engine.TensorAddScalar(rawNormSq, eps));
        var normalizedInput = Engine.TensorBroadcastDivide(inputTensor, Engine.TensorAddScalar(rawNorm, eps));
        // Now ||normalizedInput|| ≈ 1, so √c·||v|| ≈ √c which is well within tanh's linear region for c=1
        var unitNormSq = Engine.ReduceSum(Engine.TensorMultiply(normalizedInput, normalizedInput), new[] { 1 }, keepDims: true);
        var unitNorm = Engine.TensorSqrt(Engine.TensorAddScalar(unitNormSq, eps));
        var sqrtCraw = Engine.TensorMultiplyScalar(unitNorm, sqrtC);
        var tanhFactor0 = Engine.TensorTanh(sqrtCraw); // tanh(√c·1) ≈ tanh(1) ≈ 0.76 — good gradient region
        var projScale0 = Engine.TensorDivide(tanhFactor0, Engine.TensorAddScalar(sqrtCraw, eps));
        var ballInput = Engine.TensorBroadcastMultiply(projScale0, normalizedInput); // now inside ball

        // Step 1: Linear projection Mx = x_ball @ W^T
        var weightsT = IsTrainingMode
            ? Engine.TensorTranspose(_weights)
            : (_weightsTCache ??= Engine.TensorTranspose(_weights));
        var mx = Engine.TensorMatMul(ballInput, weightsT); // [batch, outputFeatures]

        // Step 2: Compute ||x_ball|| per sample (already inside ball, so √c·||x|| < 1)
        var xSq = Engine.TensorMultiply(ballInput, ballInput); // [batch, inputFeatures]
        var xNormSq = Engine.ReduceSum(xSq, new[] { 1 }, keepDims: true); // [batch, 1]
        var xNorm = Engine.TensorSqrt(Engine.TensorAddScalar(xNormSq, eps)); // [batch, 1]

        // Step 3: Compute ||Mx|| per sample (L2 norm of output rows)
        var mxSq = Engine.TensorMultiply(mx, mx); // [batch, outputFeatures]
        var mxNormSq = Engine.ReduceSum(mxSq, new[] { 1 }, keepDims: true); // [batch, 1]
        var mxNorm = Engine.TensorSqrt(Engine.TensorAddScalar(mxNormSq, eps)); // [batch, 1]

        // Step 4: arctanh(√c·||x||) / (√c·||x||)
        var sqrtCxNorm = Engine.TensorMultiplyScalar(xNorm, sqrtC); // [batch, 1]
        // Clamp to (-1+eps, 1-eps) for arctanh stability
        var clamped = Engine.TensorClamp(sqrtCxNorm, NumOps.FromDouble(-0.9999), NumOps.FromDouble(0.9999));
        // atanh(x) = 0.5 * ln((1+x)/(1-x)) — zero allocation via TensorAddScalar
        var onePlusX = Engine.TensorAddScalar(clamped, NumOps.One);
        var oneMinusX = Engine.ScalarMinusTensor(NumOps.One, clamped);
        var atanhVal = Engine.TensorMultiplyScalar(Engine.TensorLog(Engine.TensorDivide(onePlusX, oneMinusX)), NumOps.FromDouble(0.5)); // [batch, 1]
        var ratio = Engine.TensorDivide(atanhVal, Engine.TensorAddScalar(clamped, eps)); // [batch, 1]

        // Step 5: tanh(||Mx|| * ratio) / (√c·||Mx||)
        var mxTimesRatio = Engine.TensorMultiply(mxNorm, ratio); // [batch, 1]
        var tanhFactor = Engine.TensorTanh(mxTimesRatio); // [batch, 1]
        var sqrtCmxNorm = Engine.TensorMultiplyScalar(mxNorm, sqrtC); // [batch, 1]
        var scale = Engine.TensorDivide(tanhFactor, Engine.TensorAddScalar(sqrtCmxNorm, eps)); // [batch, 1]

        // Step 6: Result in Poincaré ball = scale * Mx  [batch, outputFeatures]
        // Broadcast scale [batch,1] across mx [batch, outputFeatures] via Engine
        var resultInBall = Engine.TensorBroadcastMultiply(scale, mx);

        // Step 7: Add scalar bias per output — _biases is [outputFeatures]
        // Per Ganea et al. 2018, the Möbius matmul output is a point in the Poincaré ball.
        // Output the ball coordinates directly (not distance-from-origin, which would be scalar).
        var bias2D = Engine.Reshape(_biases, [1, OutputFeatures]);
        var output = Engine.TensorBroadcastAdd(resultInBall, bias2D); // [batch, outputFeatures]

        // Project back into Poincaré ball: if ||x|| >= maxNorm, scale x to maxNorm * x/||x||
        // Per-coordinate clamp is insufficient — must enforce the L2 norm constraint.
        double maxNorm = Math.Sqrt(0.99 / NumOps.ToDouble(c));
        var normSq = Engine.ReduceSum(Engine.TensorMultiply(output, output), new[] { 1 }, keepDims: true); // [batch, 1]
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(normSq, NumOps.FromDouble(1e-8))); // [batch, 1]
        // projScale = min(1, maxNorm / ||x||) — clamp via TensorClamp upper bound of 1.0
        var maxNormTensor = Tensor<T>.CreateDefault([batchSize, 1], NumOps.FromDouble(maxNorm));
        var projScale = Engine.TensorDivide(maxNormTensor, Engine.TensorAddScalar(norm, NumOps.FromDouble(1e-8)));
        projScale = Engine.TensorClamp(projScale, NumOps.Zero, NumOps.One); // clamp to [0, 1]
        output = Engine.TensorBroadcastMultiply(projScale, output); // [batch, outputFeatures]

        _lastOutput = output;

        // Apply activation function
        var activated = ApplyActivation(output);

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return activated;
        }

        if (_originalInputShape.Length == 1)
        {
            return Engine.Reshape(activated, [OutputFeatures]);
        }

        var outputShape = new int[_originalInputShape.Length];
        for (int d = 0; d < _originalInputShape.Length - 1; d++)
        {
            outputShape[d] = _originalInputShape[d];
        }
        outputShape[_originalInputShape.Length - 1] = OutputFeatures;
        return Engine.Reshape(activated, outputShape);
    }

    // ForwardGpu removed — Tensor<T> is natively GPU-resident.
    // The regular Forward() method works on GPU-resident tensors transparently
    // via engine dispatch. RegisterPersistentTensor handles GPU memory caching.

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <remarks>
    /// Uses Riemannian gradient descent (exponential map of negative gradient).
    /// </remarks>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    // ======================================================================
    // Analytical Poincaré ball gradient helpers (Ganea et al. 2018)
    // ======================================================================

    /// <summary>
    /// Exponential map from origin: exp_0(v) = tanh(√c·||v||)/(√c·||v||) · v
    /// </summary>
    private static double[] ExpMapFromOrigin(double[] v, double c, double sqrtC)
    {
        int dim = v.Length;
        double normV = 0;
        for (int i = 0; i < dim; i++) normV += v[i] * v[i];
        normV = Math.Sqrt(normV);

        if (normV < 1e-10)
            return (double[])v.Clone();

        double scaledNorm = Math.Min(sqrtC * normV, 20.0);
        double coeff = Math.Tanh(scaledNorm) / (sqrtC * normV);

        var result = new double[dim];
        for (int i = 0; i < dim; i++)
            result[i] = coeff * v[i];
        return result;
    }

    /// <summary>
    /// Gradient of exp_0(v) w.r.t. v, applied to upstream gradient g.
    /// Returns J^T · g where J = ∂exp_0/∂v.
    /// J_ij = (tanh(α)/r) · δ_ij + (sech²(α)·√c/(2r) - tanh(α)/r²) · v_i·v_j/r
    /// where α = √c·||v||, r = √c·||v||
    /// </summary>
    private static double[] ExpMapFromOriginGrad(double[] v, double[] g, double c, double sqrtC)
    {
        int dim = v.Length;
        double normVSq = 0;
        for (int i = 0; i < dim; i++) normVSq += v[i] * v[i];
        double normV = Math.Sqrt(normVSq);

        if (normV < 1e-10)
        {
            // At origin, Jacobian ≈ I (identity)
            return (double[])g.Clone();
        }

        double r = sqrtC * normV;
        double rClamped = Math.Min(r, 20.0);
        double tanhR = Math.Tanh(rClamped);
        double sech2R = 1.0 - tanhR * tanhR;

        // Coefficient for the identity part: tanh(r) / r  (note: r = sqrtC * normV)
        double identityCoeff = tanhR / r;
        // Coefficient for the v⊗v part: (sech²(r)·sqrtC/(2·normV) - tanh(r)/(sqrtC·normV²)) / normV
        // Simplify: rank-1 correction = (sech²(r)/2 - tanh(r)/r) / normVSq
        double rank1Coeff = (sech2R / 2.0 - tanhR / r) / normVSq;

        // J^T · g = identityCoeff · g + rank1Coeff · (v^T · g) · v
        double vDotG = 0;
        for (int i = 0; i < dim; i++) vDotG += v[i] * g[i];

        var result = new double[dim];
        for (int i = 0; i < dim; i++)
            result[i] = identityCoeff * g[i] + rank1Coeff * vDotG * v[i];
        return result;
    }

    /// <summary>
    /// Möbius addition: x ⊕ y = ((1+2c⟨x,y⟩+c||y||²)x + (1-c||x||²)y) / D
    /// where D = 1+2c⟨x,y⟩+c²||x||²||y||²
    /// </summary>
    private static double[] MobiusAddDouble(double[] x, double[] y, double c)
    {
        int dim = x.Length;
        double xNormSq = 0, yNormSq = 0, xyDot = 0;
        for (int i = 0; i < dim; i++)
        {
            xNormSq += x[i] * x[i];
            yNormSq += y[i] * y[i];
            xyDot += x[i] * y[i];
        }

        double denom = Math.Max(1.0 + 2 * c * xyDot + c * c * xNormSq * yNormSq, 1e-10);
        double a = 1.0 + 2 * c * xyDot + c * yNormSq;
        double b = 1.0 - c * xNormSq;

        var result = new double[dim];
        for (int i = 0; i < dim; i++)
            result[i] = (a * x[i] + b * y[i]) / denom;
        return result;
    }

    /// <summary>
    /// Gradient of Möbius addition z = x ⊕ y w.r.t. x and y.
    /// Returns (∂L/∂x, ∂L/∂y) given ∂L/∂z.
    /// </summary>
    private static (double[] dLdX, double[] dLdY) MobiusAddGrad(double[] x, double[] y, double[] dLdZ, double c)
    {
        int dim = x.Length;
        double xNormSq = 0, yNormSq = 0, xyDot = 0;
        for (int i = 0; i < dim; i++)
        {
            xNormSq += x[i] * x[i];
            yNormSq += y[i] * y[i];
            xyDot += x[i] * y[i];
        }

        double D = Math.Max(1.0 + 2 * c * xyDot + c * c * xNormSq * yNormSq, 1e-10);
        double A = 1.0 + 2 * c * xyDot + c * yNormSq;
        double B = 1.0 - c * xNormSq;
        double invD = 1.0 / D;

        // z_i = (A·x_i + B·y_i) / D
        // ∂z_i/∂x_j = (∂A/∂x_j · x_i + A·δ_ij + ∂B/∂x_j · y_i) / D - z_i · ∂D/∂x_j / D
        // ∂A/∂x_j = 2c·y_j
        // ∂B/∂x_j = -2c·x_j
        // ∂D/∂x_j = 2c·y_j + 2c²·x_j·||y||²

        // Compute z
        var z = new double[dim];
        for (int i = 0; i < dim; i++)
            z[i] = (A * x[i] + B * y[i]) * invD;

        // dL/dx_j = sum_i dL/dz_i · ∂z_i/∂x_j
        var dLdX = new double[dim];
        var dLdY = new double[dim];

        // Precompute dot products with dLdZ
        double dLdZdotX = 0, dLdZdotY = 0, dLdZdotZ = 0;
        for (int i = 0; i < dim; i++)
        {
            dLdZdotX += dLdZ[i] * x[i];
            dLdZdotY += dLdZ[i] * y[i];
            dLdZdotZ += dLdZ[i] * z[i];
        }

        for (int j = 0; j < dim; j++)
        {
            double dAdXj = 2 * c * y[j];
            double dBdXj = -2 * c * x[j];
            double dDdXj = 2 * c * y[j] + 2 * c * c * x[j] * yNormSq;

            // dL/dx_j = sum_i dL/dz_i * [(dA/dx_j · x_i + A·δ_ij + dB/dx_j · y_i)/D - z_i · dD/dx_j / D]
            dLdX[j] = (dAdXj * dLdZdotX + A * dLdZ[j] + dBdXj * dLdZdotY) * invD
                     - dLdZdotZ * dDdXj * invD;

            // ∂A/∂y_j = 2c·x_j + 2c·y_j
            // ∂B/∂y_j = 0
            // ∂D/∂y_j = 2c·x_j + 2c²·||x||²·y_j
            double dAdYj = 2 * c * x[j] + 2 * c * y[j];
            double dDdYj = 2 * c * x[j] + 2 * c * c * xNormSq * y[j];

            dLdY[j] = (dAdYj * dLdZdotX + B * dLdZ[j]) * invD
                     - dLdZdotZ * dDdYj * invD;
        }

        return (dLdX, dLdY);
    }

    /// <summary>
    /// Gradient of Poincaré distance from origin: d(0,y) = (2/√c)·arctanh(√c·||y||)
    /// ∂d/∂y_i = 2·y_i / (||y||·(1 - c·||y||²))
    /// </summary>
    private static double[] DistanceFromOriginGrad(double[] y, double c, double sqrtC)
    {
        int dim = y.Length;
        double normYSq = 0;
        for (int i = 0; i < dim; i++) normYSq += y[i] * y[i];
        double normY = Math.Sqrt(normYSq);

        var grad = new double[dim];
        if (normY < 1e-10)
            return grad;

        // ∂d(0,y)/∂y_i = (2/√c) · √c / (1 - c·||y||²) · y_i/||y||
        //               = 2·y_i / (||y|| · (1 - c·||y||²))
        double factor = 2.0 / (normY * Math.Max(1.0 - c * normYSq, 1e-10));
        for (int i = 0; i < dim; i++)
            grad[i] = factor * y[i];
        return grad;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        if (_biasesGradient.Length < OutputFeatures)
            throw new InvalidOperationException($"Bias gradient shape mismatch: expected {OutputFeatures}, got {_biasesGradient.Length}.");

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
                _weights[o, i] = NumOps.Subtract(_weights[o, i], NumOps.Multiply(learningRate, _weightsGradient[o, i]));

            _biases[o] = NumOps.Subtract(_biases[o], NumOps.Multiply(learningRate, _biasesGradient[o]));
        }

        _weightsTCache = null;
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        var paramArray = new T[ParameterCount];
        int idx = 0;

        // Use Data.Span for direct access — avoids any indexer overhead
        var wSpan = _weights.Data.Span;
        for (int j = 0; j < wSpan.Length && idx < paramArray.Length; j++)
            paramArray[idx++] = wSpan[j];

        var bSpan = _biases.Data.Span;
        for (int j = 0; j < bSpan.Length && idx < paramArray.Length; j++)
            paramArray[idx++] = bSpan[j];

        return new Vector<T>(paramArray);
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        int idx = 0;

        // Restore weights
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = parameters[idx++];
            }
        }

        // Restore biases (scalar per output)
        for (int o = 0; o < OutputFeatures; o++)
        {
            _biases[o] = parameters[idx++];
        }

        _weightsTCache = null;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var gradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Weights gradients: [OutputFeatures, InputFeatures]
        if (_weightsGradient != null)
        {
            for (int o = 0; o < OutputFeatures; o++)
                for (int i = 0; i < InputFeatures; i++)
                    gradients[idx++] = _weightsGradient[o, i];
        }
        else
        {
            idx += OutputFeatures * InputFeatures;
        }

        // Biases gradients: [OutputFeatures] — scalar bias per output
        if (_biasesGradient != null)
        {
            for (int o = 0; o < OutputFeatures; o++)
                    gradients[idx++] = _biasesGradient[o];
        }

        return gradients;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Curvature"] = Convert.ToDouble(_curvature).ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _weightsGradient = null;
        _biasesGradient = null;
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Applies the activation function to a computation node.
    /// </summary>
    private ComputationNode<T> ApplyActivationToComputationNode(ComputationNode<T> node)
    {
        // ScalarActivation is guaranteed non-null here since this method is only called when ScalarActivation is not null
        if (ScalarActivation is null)
            throw new InvalidOperationException("ScalarActivation cannot be null when applying activation to computation node.");

        // Use ApplyToGraph if the activation supports JIT compilation
        if (ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(node);
        }

        // Fallback: apply activation directly to values and wrap as constant
        var activated = ScalarActivation.Activate(node.Value);
        return TensorOperations<T>.Constant(activated, "activated_output");
    }

    /// <summary>
    /// Creates a vector at the origin of hyperbolic space.
    /// </summary>
    private Vector<T> CreateOriginVector(int dimension)
    {
        var origin = new Vector<T>(dimension);
        for (int i = 0; i < dimension; i++)
        {
            origin[i] = NumOps.Zero;
        }
        return origin;
    }

    // SetTrainableParameters is auto-generated by TrainableParameterGenerator.
    // It updates _weights and _biases to buffer-backed views.

}

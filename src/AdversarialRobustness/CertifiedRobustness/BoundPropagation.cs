using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// A feed-forward network as a sequence of affine maps and element-wise activations, with the bound
/// propagation of interval bound propagation (Gowal et al. 2018) and CROWN (Zhang et al. 2018).
/// </summary>
/// <remarks>
/// <para>
/// Every bound is a whole-tensor engine computation generic in T. Affine weights are stored as [out, in], so
/// z = W a + b. Dense layers expose their weights as [in, out] (the library convention); a weight matrix whose
/// first axis matches the incoming width is transposed, one whose second axis matches is used as is.
/// </para>
/// <para>
/// Perturbations are bounded in L-infinity or L2 norm around the input x. The first affine map is bounded
/// exactly over the ball (radius epsilon times the dual norm of each row); later maps use boxes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
internal sealed class BoundNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private BoundNetwork(List<Operation> operations)
    {
        Operations = operations;
    }

    /// <summary>One step of the network.</summary>
    internal sealed class Operation
    {
        public Operation(Tensor<T> weight, Tensor<T> bias)
        {
            Weight = weight;
            Bias = bias;
        }

        public Operation(ActivationFunction activation)
        {
            Activation = activation;
        }

        /// <summary>The [out, in] weight of an affine step, or null for an activation.</summary>
        public Tensor<T>? Weight { get; }

        /// <summary>The [out] bias of an affine step.</summary>
        public Tensor<T>? Bias { get; }

        /// <summary>The element-wise activation of an activation step.</summary>
        public ActivationFunction Activation { get; }

        public bool IsAffine => Weight is not null;
    }

    /// <summary>The steps in forward order.</summary>
    public List<Operation> Operations { get; }

    /// <summary>
    /// Reads a network from its layers; null when a layer has weights that are not an affine matrix of the
    /// right width, so no sound bound can be computed from the layer list.
    /// </summary>
    public static BoundNetwork<T>? FromLayers(IReadOnlyList<ILayer<T>> layers, int inputLength)
    {
        var operations = new List<Operation>();
        int width = inputLength;
        foreach (var layer in layers)
        {
            var weights = layer.GetWeights();
            if (weights is not null)
            {
                var affine = AffineFrom(weights, layer.GetBiases(), width);
                if (affine?.Weight is null)
                {
                    return null;
                }

                operations.Add(affine);
                width = affine.Weight._shape[0];
            }
            else if (ShapeLength(layer.GetOutputShape()) != width)
            {
                // A parameter-free layer that changes the width (pooling, cropping, ...) is not element-wise.
                return null;
            }

            foreach (var activation in layer.GetActivationTypes())
            {
                if (activation == ActivationFunction.Identity || activation == ActivationFunction.Linear)
                {
                    continue;
                }

                if (!IsSupported(activation))
                {
                    return null;
                }

                operations.Add(new Operation(activation));
            }
        }

        return new BoundNetwork<T>(operations);
    }

    /// <summary>
    /// Reads the layers of a neural network model (the network's own layer list, else its architecture's);
    /// null for any other model or when the layers cannot be bounded.
    /// </summary>
    public static BoundNetwork<T>? FromModel(object model, int inputLength)
    {
        IReadOnlyList<ILayer<T>>? layers = null;
        if (model is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> network && network.Layers.Count > 0)
        {
            layers = network.Layers;
        }
        else if (model is INeuralNetworkModel<T> neuralModel)
        {
            var architectureLayers = neuralModel.GetArchitecture().Layers;
            if (architectureLayers is not null && architectureLayers.Count > 0)
            {
                layers = architectureLayers;
            }
        }

        return layers is null ? null : FromLayers(layers, inputLength);
    }

    /// <summary>Width of the network output.</summary>
    public int OutputLength(int inputLength)
    {
        int width = inputLength;
        foreach (var operation in Operations)
        {
            width = operation.Weight is null ? width : operation.Weight._shape[0];
        }

        return width;
    }

    /// <summary>The activations with sound interval and linear bounds here.</summary>
    internal static bool IsSupported(ActivationFunction activation) =>
        activation == ActivationFunction.ReLU
        || activation == ActivationFunction.LeakyReLU
        || activation == ActivationFunction.Sigmoid
        || activation == ActivationFunction.Tanh
        || activation == ActivationFunction.Softmax;

    private static int ShapeLength(int[] shape)
    {
        int length = 1;
        foreach (int dimension in shape)
        {
            // Unresolved (-1) and empty placeholders carry no width.
            length *= dimension > 0 ? dimension : 1;
        }

        return length;
    }

    private static Operation? AffineFrom(Tensor<T> weights, Tensor<T>? biases, int width)
    {
        var engine = AiDotNetEngine.Current;
        if (weights.Rank != 2)
        {
            return null;
        }

        var shape = weights._shape;
        Tensor<T> outByIn;
        if (shape[0] == width)
        {
            // The library's dense layers store [in, out].
            outByIn = engine.TensorTranspose(weights);
        }
        else if (shape[1] == width)
        {
            outByIn = weights;
        }
        else
        {
            return null;
        }

        int outputs = outByIn._shape[0];
        Tensor<T> bias;
        if (biases is null || biases.Length == 0)
        {
            bias = new Tensor<T>(new[] { outputs });
        }
        else if (biases.Length == outputs)
        {
            bias = engine.Reshape(biases, new[] { outputs });
        }
        else
        {
            return null;
        }

        return new Operation(outByIn, bias);
    }

    #region Shared tensor helpers

    private static Tensor<T> MatVec(Tensor<T> matrix, Tensor<T> vector)
    {
        var engine = AiDotNetEngine.Current;
        int rows = matrix._shape[0];
        var column = engine.Reshape(vector, new[] { vector.Length, 1 });
        return engine.Reshape(engine.TensorMatMul(matrix, column), new[] { rows });
    }

    /// <summary>Row-wise dual norm: L1 for an L-infinity ball, L2 for an L2 ball.</summary>
    private static Tensor<T> DualRowNorms(Tensor<T> matrix, AdversarialNormType norm)
    {
        var engine = AiDotNetEngine.Current;
        return norm == AdversarialNormType.L2
            ? engine.TensorNorm(matrix, 1, false)
            : engine.ReduceSum(engine.TensorAbs(matrix), new[] { 1 }, false);
    }

    /// <summary>
    /// Lower bound of rows(A) . x' + bias over the ball around x: A x + bias - epsilon ||A_s||_dual.
    /// </summary>
    private static Tensor<T> LowerOverBall(Tensor<T> coefficients, Tensor<T> bias, Tensor<T> x, T epsilon, AdversarialNormType norm)
    {
        var engine = AiDotNetEngine.Current;
        var center = engine.TensorAdd(MatVec(coefficients, x), bias);
        return engine.TensorSubtract(center, engine.TensorMultiplyScalar(DualRowNorms(coefficients, norm), epsilon));
    }

    /// <summary>
    /// Interval of W a + b over the box [lower, upper]: center W c + b, radius |W| r.
    /// </summary>
    private static (Tensor<T> Lower, Tensor<T> Upper) AffineOverBox(Tensor<T> weight, Tensor<T> bias, Tensor<T> lower, Tensor<T> upper)
    {
        var engine = AiDotNetEngine.Current;
        T half = NumOps.FromDouble(0.5);
        var center = engine.TensorMultiplyScalar(engine.TensorAdd(lower, upper), half);
        var radius = engine.TensorMultiplyScalar(engine.TensorSubtract(upper, lower), half);
        var mid = engine.TensorAdd(MatVec(weight, center), bias);
        var spread = MatVec(engine.TensorAbs(weight), radius);
        return (engine.TensorSubtract(mid, spread), engine.TensorAdd(mid, spread));
    }

    /// <summary>
    /// Interval of W x' + b over the ball: W x + b -/+ epsilon times the dual norm of each row.
    /// </summary>
    private static (Tensor<T> Lower, Tensor<T> Upper) AffineOverBall(Tensor<T> weight, Tensor<T> bias, Tensor<T> x, T epsilon, AdversarialNormType norm)
    {
        var engine = AiDotNetEngine.Current;
        var mid = engine.TensorAdd(MatVec(weight, x), bias);
        var spread = engine.TensorMultiplyScalar(DualRowNorms(weight, norm), epsilon);
        return (engine.TensorSubtract(mid, spread), engine.TensorAdd(mid, spread));
    }

    /// <summary>
    /// The interval image of a monotone element-wise activation; softmax outputs lie in [0, 1].
    /// </summary>
    private static (Tensor<T> Lower, Tensor<T> Upper) ActivationOverBox(ActivationFunction activation, Tensor<T> lower, Tensor<T> upper)
    {
        var engine = AiDotNetEngine.Current;
        switch (activation)
        {
            case ActivationFunction.ReLU:
                return (engine.ReLU(lower), engine.ReLU(upper));
            case ActivationFunction.Sigmoid:
                return (engine.TensorSigmoid(lower), engine.TensorSigmoid(upper));
            case ActivationFunction.Tanh:
                return (engine.TensorTanh(lower), engine.TensorTanh(upper));
            case ActivationFunction.LeakyReLU:
                return (LeakyReLU(lower), LeakyReLU(upper));
            case ActivationFunction.Softmax:
                return (engine.TensorZerosLike(lower), engine.TensorOnesLike(upper));
            default:
                return (lower, upper);
        }
    }

    /// <summary>The default LeakyReLU slope of the library's activation (0.01).</summary>
    internal static T LeakySlope => NumOps.FromDouble(0.01);

    private static Tensor<T> LeakyReLU(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMax(values, engine.TensorMultiplyScalar(values, LeakySlope));
    }

    #endregion

    #region Interval bound propagation (Gowal et al. 2018)

    /// <summary>
    /// Index of the last affine step when every step after it preserves the arg-max of the logits it
    /// produces (identity, softmax and strictly increasing activations, or ReLU when the top logit stays
    /// positive), else -1.
    /// </summary>
    private int LogitStep(out bool requiresPositiveTop)
    {
        requiresPositiveTop = false;
        int last = Operations.FindLastIndex(op => op.IsAffine);
        if (last < 0)
        {
            return -1;
        }

        for (int i = last + 1; i < Operations.Count; i++)
        {
            requiresPositiveTop |= Operations[i].Activation == ActivationFunction.ReLU;
        }

        return last;
    }

    /// <summary>
    /// IBP bounds on the output of steps [0, count) over the ball around x.
    /// </summary>
    public (Tensor<T> Lower, Tensor<T> Upper) IntervalBounds(Tensor<T> x, T epsilon, AdversarialNormType norm, int count)
    {
        var engine = AiDotNetEngine.Current;
        var lower = engine.TensorAddScalar(x, NumOps.Negate(epsilon));
        var upper = engine.TensorAddScalar(x, epsilon);
        bool atInput = true;
        for (int i = 0; i < count; i++)
        {
            (lower, upper) = IntervalStep(Operations[i], x, epsilon, norm, lower, upper, atInput);
            atInput = false;
        }

        return (lower, upper);
    }

    private static (Tensor<T> Lower, Tensor<T> Upper) IntervalStep(
        Operation operation,
        Tensor<T> x,
        T epsilon,
        AdversarialNormType norm,
        Tensor<T> lower,
        Tensor<T> upper,
        bool atInput)
    {
        if (operation.Weight is not null && operation.Bias is not null)
        {
            return atInput
                ? AffineOverBall(operation.Weight, operation.Bias, x, epsilon, norm)
                : AffineOverBox(operation.Weight, operation.Bias, lower, upper);
        }

        return ActivationOverBox(operation.Activation, lower, upper);
    }

    /// <summary>
    /// IBP lower bounds of the logit margins z_y - z_j, with the margin specification merged into the last
    /// affine layer (Gowal et al. 2018, Sec. 3); the entry for j = y is positive infinity in effect (the
    /// largest value of T). Also returns whether the top logit must stay positive (a trailing ReLU).
    /// </summary>
    public Tensor<T> IntervalMargins(Tensor<T> x, T epsilon, AdversarialNormType norm, int label, out Tensor<T> topLogitLower)
    {
        var engine = AiDotNetEngine.Current;
        int step = LogitStep(out bool requiresPositiveTop);
        if (step < 0)
        {
            var (lower, upper) = IntervalBounds(x, epsilon, norm, Operations.Count);
            topLogitLower = engine.TensorSlice(lower, new[] { label }, new[] { 1 });
            return ExcludeLabel(engine.TensorSubtract(engine.TensorAddScalar(engine.TensorZerosLike(upper), At(lower, label)), upper), label);
        }

        var logitLayer = Operations[step];
        var (spec, specBias) = MarginSpecification(logitLayer, label);
        Tensor<T> margins;
        if (step == 0)
        {
            margins = LowerOverBall(spec, specBias, x, epsilon, norm);
            topLogitLower = LowerOverBall(Row(logitLayer, label), Entry(logitLayer, label), x, epsilon, norm);
        }
        else
        {
            var (lower, upper) = IntervalBounds(x, epsilon, norm, step);
            margins = AffineOverBox(spec, specBias, lower, upper).Lower;
            topLogitLower = AffineOverBox(Row(logitLayer, label), Entry(logitLayer, label), lower, upper).Lower;
        }

        if (!requiresPositiveTop)
        {
            topLogitLower = engine.TensorAddScalar(engine.TensorZerosLike(topLogitLower), NumOps.One);
        }

        return ExcludeLabel(margins, label);
    }

    /// <summary>
    /// The rows (e_y - e_j)^T W and entries (e_y - e_j)^T b of the margin specification merged with an affine layer.
    /// </summary>
    private static (Tensor<T> Weight, Tensor<T> Bias) MarginSpecification(Operation layer, int label)
    {
        var engine = AiDotNetEngine.Current;
        var weight = layer.Weight ?? new Tensor<T>(new[] { 0, 0 });
        var bias = layer.Bias ?? new Tensor<T>(new[] { 0 });
        int outputs = weight._shape[0];
        int inputs = weight._shape[1];
        var labelRow = engine.TensorSlice(weight, new[] { label, 0 }, new[] { 1, inputs });
        var specWeight = engine.TensorSubtract(engine.TensorBroadcastTo(labelRow, new[] { outputs, inputs }), weight);
        var specBias = engine.TensorSubtract(engine.TensorAddScalar(engine.TensorZerosLike(bias), At(bias, label)), bias);
        return (specWeight, specBias);
    }

    private static Tensor<T> Row(Operation layer, int index)
    {
        var weight = layer.Weight ?? new Tensor<T>(new[] { 0, 0 });
        return AiDotNetEngine.Current.TensorSlice(weight, new[] { index, 0 }, new[] { 1, weight._shape[1] });
    }

    private static Tensor<T> Entry(Operation layer, int index)
    {
        var bias = layer.Bias ?? new Tensor<T>(new[] { 0 });
        return AiDotNetEngine.Current.TensorSlice(bias, new[] { index }, new[] { 1 });
    }

    private static T At(Tensor<T> tensor, int index) => tensor.GetFlat(index);

    /// <summary>Replaces the margin of the label against itself by the largest value of T.</summary>
    private static Tensor<T> ExcludeLabel(Tensor<T> margins, int label)
    {
        var engine = AiDotNetEngine.Current;
        var isLabel = Vector<T>.CreateStandardBasis(margins.Length, label).AsTensor();
        var largest = engine.TensorAddScalar(engine.TensorZerosLike(margins), NumOps.MaxValue);
        return engine.TensorWhere(isLabel, largest, margins);
    }

    #endregion

    #region CROWN (Zhang et al. 2018)

    /// <summary>
    /// Whether CROWN's element-wise linear relaxations apply: softmax may only follow the last affine step.
    /// </summary>
    public bool SupportsCrown
    {
        get
        {
            int last = Operations.FindLastIndex(op => op.IsAffine);
            for (int i = 0; i < Operations.Count; i++)
            {
                if (!Operations[i].IsAffine && Operations[i].Activation == ActivationFunction.Softmax && i < last)
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <summary>
    /// CROWN lower bounds of the logit margins z_y - z_j (the entry for j = y is the largest value of T),
    /// and of the top logit z_y.
    /// </summary>
    /// <remarks>
    /// Pre-activation bounds of every hidden layer are computed layer by layer with CROWN itself
    /// (Zhang et al. 2018, Sec. 3.1, Theorem 3.2); each activation input bound uses the linear relaxations
    /// of all earlier activations, so later layers get bounds at least as tight as IBP's.
    /// </remarks>
    public Tensor<T> CrownMargins(Tensor<T> x, T epsilon, AdversarialNormType norm, int label, out Tensor<T> topLogitLower)
    {
        var engine = AiDotNetEngine.Current;
        int step = LogitStep(out bool requiresPositiveTop);
        var inputBounds = CrownActivationInputBounds(x, epsilon, norm);
        if (step < 0)
        {
            var (lower, upper) = IntervalBounds(x, epsilon, norm, Operations.Count);
            topLogitLower = engine.TensorSlice(lower, new[] { label }, new[] { 1 });
            return ExcludeLabel(engine.TensorSubtract(engine.TensorAddScalar(engine.TensorZerosLike(upper), At(lower, label)), upper), label);
        }

        int outputs = Operations[step].Weight?._shape[0] ?? 0;
        var labelRow = Vector<T>.CreateStandardBasis(outputs, label).AsTensor();
        var spec = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(labelRow, new[] { 1, outputs }), new[] { outputs, outputs }),
            CreateIdentity(outputs));
        var margins = CrownLower(spec, step, x, epsilon, norm, inputBounds);
        topLogitLower = requiresPositiveTop
            ? CrownLower(engine.Reshape(labelRow, new[] { 1, outputs }), step, x, epsilon, norm, inputBounds)
            : engine.TensorAddScalar(new Tensor<T>(new[] { 1 }), NumOps.One);
        return ExcludeLabel(margins, label);
    }

    /// <summary>
    /// CROWN bounds on every output of the network's last affine step (for reporting).
    /// </summary>
    public (Tensor<T> Lower, Tensor<T> Upper) CrownLogitBounds(Tensor<T> x, T epsilon, AdversarialNormType norm)
    {
        int step = LogitStep(out _);
        if (step < 0)
        {
            return IntervalBounds(x, epsilon, norm, Operations.Count);
        }

        var inputBounds = CrownActivationInputBounds(x, epsilon, norm);
        return CrownBoundsOf(step, x, epsilon, norm, inputBounds);
    }

    private static Tensor<T> CreateIdentity(int size) => Tensor<T>.CreateIdentity(size);

    /// <summary>
    /// Bounds on the input of every activation step, in forward order (null for affine steps).
    /// </summary>
    private List<(Tensor<T> Lower, Tensor<T> Upper)?> CrownActivationInputBounds(Tensor<T> x, T epsilon, AdversarialNormType norm)
    {
        var engine = AiDotNetEngine.Current;
        var bounds = new List<(Tensor<T> Lower, Tensor<T> Upper)?>();
        (Tensor<T> Lower, Tensor<T> Upper) current = (engine.TensorAddScalar(x, NumOps.Negate(epsilon)), engine.TensorAddScalar(x, epsilon));
        for (int i = 0; i < Operations.Count; i++)
        {
            var operation = Operations[i];
            if (operation.IsAffine)
            {
                bounds.Add(null);
                current = CrownBoundsOf(i, x, epsilon, norm, bounds);
            }
            else
            {
                bounds.Add(current);
                current = ActivationOverBox(operation.Activation, current.Lower, current.Upper);
            }
        }

        return bounds;
    }

    /// <summary>
    /// CROWN lower and upper bounds of every output of affine step <paramref name="step"/>.
    /// </summary>
    private (Tensor<T> Lower, Tensor<T> Upper) CrownBoundsOf(
        int step,
        Tensor<T> x,
        T epsilon,
        AdversarialNormType norm,
        List<(Tensor<T> Lower, Tensor<T> Upper)?> inputBounds)
    {
        var engine = AiDotNetEngine.Current;
        int outputs = Operations[step].Weight?._shape[0] ?? 0;
        var identity = CreateIdentity(outputs);
        var lower = CrownLower(identity, step, x, epsilon, norm, inputBounds);
        var upper = engine.TensorNegate(CrownLower(engine.TensorNegate(identity), step, x, epsilon, norm, inputBounds));
        return (lower, upper);
    }

    /// <summary>
    /// Lower bound of spec . z over the ball, where z is the output of affine step <paramref name="step"/>, by
    /// back-substituting the affine steps and the activations' linear relaxations (Zhang et al. 2018, Thm. 3.2).
    /// </summary>
    private Tensor<T> CrownLower(
        Tensor<T> spec,
        int step,
        Tensor<T> x,
        T epsilon,
        AdversarialNormType norm,
        List<(Tensor<T> Lower, Tensor<T> Upper)?> inputBounds)
    {
        var engine = AiDotNetEngine.Current;
        var coefficients = spec;
        var bias = new Tensor<T>(new[] { spec._shape[0] });
        for (int i = step; i >= 0; i--)
        {
            var operation = Operations[i];
            if (operation.Weight is not null && operation.Bias is not null)
            {
                bias = engine.TensorAdd(bias, MatVec(coefficients, operation.Bias));
                coefficients = engine.TensorMatMul(coefficients, operation.Weight);
            }
            else if (i < inputBounds.Count && inputBounds[i] is { } box)
            {
                (coefficients, bias) = BackSubstituteActivation(coefficients, bias, operation.Activation, box.Lower, box.Upper);
            }
        }

        return LowerOverBall(coefficients, bias, x, epsilon, norm);
    }

    /// <summary>
    /// Replaces a = sigma(z) in A a + bias by its linear relaxation: positive coefficients take the lower line,
    /// negative ones the upper line.
    /// </summary>
    private static (Tensor<T> Coefficients, Tensor<T> Bias) BackSubstituteActivation(
        Tensor<T> coefficients,
        Tensor<T> bias,
        ActivationFunction activation,
        Tensor<T> lower,
        Tensor<T> upper)
    {
        var engine = AiDotNetEngine.Current;
        var relaxation = Relax(activation, lower, upper);
        var positive = engine.TensorMax(coefficients, NumOps.Zero);
        var negative = engine.TensorMin(coefficients, NumOps.Zero);
        var newBias = engine.TensorAdd(bias, engine.TensorAdd(MatVec(positive, relaxation.LowerIntercept), MatVec(negative, relaxation.UpperIntercept)));
        int rows = coefficients._shape[0];
        int columns = coefficients._shape[1];
        var lowerSlopes = engine.TensorBroadcastTo(engine.Reshape(relaxation.LowerSlope, new[] { 1, columns }), new[] { rows, columns });
        var upperSlopes = engine.TensorBroadcastTo(engine.Reshape(relaxation.UpperSlope, new[] { 1, columns }), new[] { rows, columns });
        var newCoefficients = engine.TensorAdd(engine.TensorMultiply(positive, lowerSlopes), engine.TensorMultiply(negative, upperSlopes));
        return (newCoefficients, newBias);
    }

    /// <summary>Linear lower and upper bounds alpha z + beta of an activation on [l, u], per neuron.</summary>
    internal readonly struct Relaxation
    {
        public Relaxation(Tensor<T> lowerSlope, Tensor<T> lowerIntercept, Tensor<T> upperSlope, Tensor<T> upperIntercept)
        {
            LowerSlope = lowerSlope;
            LowerIntercept = lowerIntercept;
            UpperSlope = upperSlope;
            UpperIntercept = upperIntercept;
        }

        public Tensor<T> LowerSlope { get; }

        public Tensor<T> LowerIntercept { get; }

        public Tensor<T> UpperSlope { get; }

        public Tensor<T> UpperIntercept { get; }
    }

    /// <summary>
    /// CROWN's linear relaxations (Zhang et al. 2018, Sec. 3.2 and Table 1).
    /// </summary>
    internal static Relaxation Relax(ActivationFunction activation, Tensor<T> lower, Tensor<T> upper)
    {
        switch (activation)
        {
            case ActivationFunction.ReLU:
                return PiecewiseLinear(lower, upper, NumOps.Zero);
            case ActivationFunction.LeakyReLU:
                return PiecewiseLinear(lower, upper, LeakySlope);
            case ActivationFunction.Sigmoid:
            case ActivationFunction.Tanh:
                return Smooth(activation, lower, upper);
            default:
                var engine = AiDotNetEngine.Current;
                return new Relaxation(engine.TensorOnesLike(lower), engine.TensorZerosLike(lower), engine.TensorOnesLike(lower), engine.TensorZerosLike(lower));
        }
    }

    /// <summary>
    /// max(x, a x) with a in [0, 1): active neurons pass through, inactive ones scale by a; a crossing neuron
    /// gets the chord as upper line and, as CROWN-Ada, the lower line through the origin with slope 1 when
    /// u >= -l and a otherwise.
    /// </summary>
    private static Relaxation PiecewiseLinear(Tensor<T> lower, Tensor<T> upper, T slope)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.TensorZerosLike(lower);
        var ones = engine.TensorOnesLike(lower);
        var slopes = engine.TensorAddScalar(zeros, slope);
        var active = engine.TensorWhere(engine.TensorLessThan(lower, NumOps.Zero), zeros, ones);
        var inactive = engine.TensorWhere(engine.TensorGreaterThan(upper, NumOps.Zero), zeros, ones);

        // Chord through (l, a l) and (u, u): slope (u - a l) / (u - l), intercept u - slope * u.
        var width = engine.TensorSubtract(upper, lower);
        var safeWidth = engine.TensorWhere(engine.TensorGreaterThan(width, NumOps.Zero), width, ones);
        var chordSlope = engine.TensorDivide(engine.TensorSubtract(upper, engine.TensorMultiplyScalar(lower, slope)), safeWidth);
        var chordIntercept = engine.TensorSubtract(upper, engine.TensorMultiply(chordSlope, upper));
        var adaptive = engine.TensorWhere(engine.TensorLessThan(engine.TensorAdd(upper, lower), NumOps.Zero), slopes, ones);

        var upperSlope = engine.TensorWhere(active, ones, engine.TensorWhere(inactive, slopes, chordSlope));
        var upperIntercept = engine.TensorWhere(active, zeros, engine.TensorWhere(inactive, zeros, chordIntercept));
        var lowerSlope = engine.TensorWhere(active, ones, engine.TensorWhere(inactive, slopes, adaptive));
        return new Relaxation(lowerSlope, zeros, upperSlope, upperIntercept);
    }

    private static Tensor<T> Apply(ActivationFunction activation, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return activation == ActivationFunction.Tanh ? engine.TensorTanh(values) : engine.TensorSigmoid(values);
    }

    /// <summary>sigmoid' = s (1 - s); tanh' = 1 - t^2.</summary>
    private static Tensor<T> Derivative(ActivationFunction activation, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var f = Apply(activation, values);
        return activation == ActivationFunction.Tanh
            ? engine.ScalarMinusTensor(NumOps.One, engine.TensorSquare(f))
            : engine.TensorMultiply(f, engine.ScalarMinusTensor(NumOps.One, f));
    }

    /// <summary>
    /// Sigmoid and tanh, convex on (-inf, 0] and concave on [0, inf): chord and mid-point tangent on one-sided
    /// intervals; on crossing intervals the tangent through the far endpoint (found by bisection), or the chord
    /// when that tangent point lies outside the interval.
    /// </summary>
    private static Relaxation Smooth(ActivationFunction activation, Tensor<T> lower, Tensor<T> upper)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.TensorZerosLike(lower);
        var ones = engine.TensorOnesLike(lower);
        var fl = Apply(activation, lower);
        var fu = Apply(activation, upper);
        var width = engine.TensorSubtract(upper, lower);
        var positiveWidth = engine.TensorGreaterThan(width, NumOps.Zero);
        var safeWidth = engine.TensorWhere(positiveWidth, width, ones);
        var chordSlope = engine.TensorWhere(positiveWidth, engine.TensorDivide(engine.TensorSubtract(fu, fl), safeWidth), Derivative(activation, lower));
        var chordIntercept = engine.TensorSubtract(fl, engine.TensorMultiply(chordSlope, lower));

        var mid = engine.TensorMultiplyScalar(engine.TensorAdd(lower, upper), NumOps.FromDouble(0.5));
        var (midSlope, midIntercept) = Tangent(activation, mid);

        // Crossing: upper tangent at d in [0, u] through (l, f(l)); lower tangent at d in [l, 0] through (u, f(u)).
        var upperPoint = TangentThrough(activation, zeros, upper, lower, fl, true);
        var lowerPoint = TangentThrough(activation, lower, zeros, upper, fu, false);
        var (upperTanSlope, upperTanIntercept) = Tangent(activation, upperPoint.Point);
        var (lowerTanSlope, lowerTanIntercept) = Tangent(activation, lowerPoint.Point);
        var crossUpperSlope = engine.TensorWhere(upperPoint.Found, upperTanSlope, chordSlope);
        var crossUpperIntercept = engine.TensorWhere(upperPoint.Found, upperTanIntercept, chordIntercept);
        var crossLowerSlope = engine.TensorWhere(lowerPoint.Found, lowerTanSlope, chordSlope);
        var crossLowerIntercept = engine.TensorWhere(lowerPoint.Found, lowerTanIntercept, chordIntercept);

        var convex = engine.TensorWhere(engine.TensorGreaterThan(upper, NumOps.Zero), zeros, ones);   // u <= 0
        var concave = engine.TensorWhere(engine.TensorLessThan(lower, NumOps.Zero), zeros, ones);     // l >= 0

        var upperSlope = engine.TensorWhere(convex, chordSlope, engine.TensorWhere(concave, midSlope, crossUpperSlope));
        var upperIntercept = engine.TensorWhere(convex, chordIntercept, engine.TensorWhere(concave, midIntercept, crossUpperIntercept));
        var lowerSlope = engine.TensorWhere(convex, midSlope, engine.TensorWhere(concave, chordSlope, crossLowerSlope));
        var lowerIntercept = engine.TensorWhere(convex, midIntercept, engine.TensorWhere(concave, chordIntercept, crossLowerIntercept));
        return new Relaxation(lowerSlope, lowerIntercept, upperSlope, upperIntercept);
    }

    private static (Tensor<T> Slope, Tensor<T> Intercept) Tangent(ActivationFunction activation, Tensor<T> point)
    {
        var engine = AiDotNetEngine.Current;
        var slope = Derivative(activation, point);
        var intercept = engine.TensorSubtract(Apply(activation, point), engine.TensorMultiply(slope, point));
        return (slope, intercept);
    }

    /// <summary>
    /// Bisects, for every neuron at once, the point d in [from, to] whose tangent passes through (anchor, f(anchor)).
    /// g(d) = f(d) + f'(d)(anchor - d) - f(anchor) is monotone on the interval; Found is 1 where it changes sign.
    /// </summary>
    private static (Tensor<T> Point, Tensor<T> Found) TangentThrough(
        ActivationFunction activation,
        Tensor<T> from,
        Tensor<T> to,
        Tensor<T> anchor,
        Tensor<T> anchorValue,
        bool increasing)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.TensorZerosLike(from);
        var ones = engine.TensorOnesLike(from);
        var gFrom = Residual(activation, from, anchor, anchorValue);
        var gTo = Residual(activation, to, anchor, anchorValue);

        // g' = f''(d)(anchor - d) > 0 on both searches, so g increases. The upper line needs g(d) >= 0 (it stays
        // above f(l)), the lower line g(d) <= 0 (it stays below f(u)); a root exists in the interval iff g(u) >= 0
        // for the upper search and g(l) <= 0 for the lower one.
        var found = increasing
            ? engine.TensorWhere(engine.TensorLessThan(gTo, NumOps.Zero), zeros, ones)
            : engine.TensorWhere(engine.TensorGreaterThan(gFrom, NumOps.Zero), zeros, ones);
        var low = from;
        var high = to;
        for (int iteration = 0; iteration < 30; iteration++)
        {
            (low, high) = BisectStep(activation, low, high, anchor, anchorValue, increasing);
        }

        // Bisection keeps g(low) < 0 <= g(high): the upper line takes high, the lower line low (both sound).
        var point = increasing ? high : low;
        return (point, found);
    }

    private static (Tensor<T> Low, Tensor<T> High) BisectStep(
        ActivationFunction activation,
        Tensor<T> low,
        Tensor<T> high,
        Tensor<T> anchor,
        Tensor<T> anchorValue,
        bool increasing)
    {
        var engine = AiDotNetEngine.Current;
        var mid = engine.TensorMultiplyScalar(engine.TensorAdd(low, high), NumOps.FromDouble(0.5));
        var g = Residual(activation, mid, anchor, anchorValue);

        // g increases, so g(mid) < 0 puts the root right of mid.
        var rootRight = engine.TensorLessThan(g, NumOps.Zero);
        return (engine.TensorWhere(rootRight, mid, low), engine.TensorWhere(rootRight, high, mid));
    }

    private static Tensor<T> Residual(ActivationFunction activation, Tensor<T> point, Tensor<T> anchor, Tensor<T> anchorValue)
    {
        var engine = AiDotNetEngine.Current;
        var tangentAtAnchor = engine.TensorAdd(
            Apply(activation, point),
            engine.TensorMultiply(Derivative(activation, point), engine.TensorSubtract(anchor, point)));
        return engine.TensorSubtract(tangentAtAnchor, anchorValue);
    }

    #endregion
}

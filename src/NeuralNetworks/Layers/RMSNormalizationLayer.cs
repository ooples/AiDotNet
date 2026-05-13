using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Root-Mean-Square Layer Normalization (Zhang &amp; Sennrich 2019).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RMSNorm rescales each sample's feature vector by its root-mean-square magnitude
/// (without re-centering on the mean), then multiplies by a learnable per-feature
/// gain γ. Concretely:
/// </para>
/// <para>
/// <c>RMSNorm(x)_i = (x_i / sqrt(mean(x²) + ε)) · γ_i</c>
/// </para>
/// <para>
/// This is the normalization used by T5 (Raffel et al. 2020), LLaMA (Touvron et al.
/// 2023), Gemma (Gemma Team 2024), Qwen2 (Yang et al. 2024), and ChatGLM3
/// (Zeng et al. 2023). Unlike standard LayerNorm there is NO learnable shift β —
/// the paper-canonical RMSNorm formulation only includes a multiplicative gain.
/// </para>
/// <para><b>For Beginners:</b> RMSNorm is a simpler, faster cousin of LayerNorm.
///
/// LayerNorm: subtracts the mean, divides by standard deviation, then scales and shifts.
/// RMSNorm: skips the mean-subtraction step, divides by the root-mean-square, then scales.
///
/// The "skip the mean" part isn't a shortcut — Zhang &amp; Sennrich showed it works
/// just as well in practice but costs less compute. Every modern LLM-style text
/// encoder (T5, LLaMA, Gemma, Qwen, ChatGLM) uses RMSNorm because it's both
/// paper-validated and faster.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang &amp; Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, HasTrainingMode = false, TestInputShape = "1, 4", TestConstructorArgs = "")]
public partial class RMSNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;

    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _gamma;

    private Tensor<T>? _gammaGradient;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the gain (γ) parameters of the layer.
    /// </summary>
    public Vector<T> GetGamma() => _gamma.ToVector();

    /// <summary>
    /// Gets the gain (γ) tensor for JIT compilation and internal use.
    /// </summary>
    public Tensor<T> GetGammaTensor() => _gamma;

    /// <summary>
    /// Gets the epsilon value used for numerical stability.
    /// </summary>
    public T GetEpsilon() => _epsilon;

    /// <summary>
    /// Returns layer-specific metadata required for cloning/serialization.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Epsilon"] = Convert.ToDouble(_epsilon, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RMSNormalizationLayer{T}"/> class.
    /// </summary>
    /// <param name="epsilon">A small value added to the mean-square for numerical stability.
    /// Default 1e-6 matches the T5 / LLaMA / Gemma / Qwen2 / ChatGLM3 paper convention
    /// (LayerNorm uses 1e-5, but RMSNorm without mean-subtraction tolerates a tighter
    /// epsilon).</param>
    /// <remarks>
    /// Feature size is resolved lazily on the first forward pass from
    /// <c>input.Shape[^1]</c>, matching the <see cref="LayerNormalizationLayer{T}"/>
    /// convention.
    /// </remarks>
    public RMSNormalizationLayer(double epsilon = 1e-6)
        : base(new[] { -1 }, new[] { -1 })
    {
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        // Lazy init: feature dim resolved from input.Shape[^1] on first forward.
        _gamma = new Tensor<T>([0]);
    }

    /// <summary>
    /// Resolves <c>featureSize</c> from <c>input.Shape[^1]</c> on the first forward call
    /// and allocates the γ tensor. Per Zhang &amp; Sennrich 2019, γ is initialised to 1
    /// (the no-op identity) so the layer behaves as a pass-through before training.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int featureSize = input.Shape[input.Shape.Length - 1];
        if (featureSize <= 0)
        {
            throw new ArgumentException(
                $"RMSNormalizationLayer cannot resolve featureSize: input.Shape[^1] = {featureSize}.",
                nameof(input));
        }

        _gamma = Tensor<T>.CreateDefault([featureSize], NumOps.One);
        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);

        ResolveShapes(new[] { featureSize }, new[] { featureSize });
    }

    /// <summary>
    /// Performs the forward pass of RMSNorm via composed Engine ops so the gradient
    /// tape records each step and Backward is supplied automatically by autodiff —
    /// no manual backward kernel needed.
    /// </summary>
    /// <remarks>
    /// Compute graph (last axis normalisation):
    /// <list type="number">
    /// <item><c>x² = TensorSquare(x)</c></item>
    /// <item><c>sumSq = ReduceSum(x², axes=[last], keepDims=true)</c></item>
    /// <item><c>meanSq = sumSq / featureSize</c></item>
    /// <item><c>rms = sqrt(meanSq + ε)</c></item>
    /// <item><c>invRms = 1 / rms</c></item>
    /// <item><c>normalised = x · invRms</c> (broadcast over the last axis)</item>
    /// <item><c>output = normalised · γ</c> (broadcast over the last axis)</item>
    /// </list>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        int rank = input.Shape.Length;
        int featureSize = input.Shape[rank - 1];

        // mean(x²) along the last axis, keeping that axis as size-1 so broadcast
        // shape inference treats the result as [..., 1] against the original [..., F].
        var squared = Engine.TensorSquare(input);
        var sumSq = Engine.ReduceSum(squared, new[] { rank - 1 }, keepDims: true);
        var meanSq = Engine.TensorDivideScalar(sumSq, NumOps.FromDouble(featureSize));

        // Add ε before sqrt so a zero-input doesn't divide by zero.
        var stabilised = Engine.TensorAddScalar(meanSq, _epsilon);
        var rms = Engine.TensorSqrt(stabilised);
        var invRms = Engine.TensorReciprocal(rms);

        // x · (1/rms), then apply per-feature γ. Both broadcast along the last axis.
        var normalised = Engine.TensorBroadcastMultiply(input, invRms);
        var output = Engine.TensorBroadcastMultiply(normalised, _gamma);

        return output;
    }

    public override long ParameterCount => _gamma.Length;

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _gamma.ToVector();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Round-trip from saved parameters when the layer is still in lazy
        // placeholder state. RMSNorm has only γ (no β), so featureSize equals
        // parameters.Length directly.
        if (!IsShapeResolved)
        {
            if (parameters.Length == 0) return;
            ResolveFromShape(new[] { parameters.Length });
        }

        if (parameters.Length != _gamma.Length)
        {
            throw new ArgumentException(
                $"Expected {_gamma.Length} parameters, but got {parameters.Length}.");
        }

        var gSpan = _gamma.Data.Span;
        for (int i = 0; i < _gamma.Length; i++) gSpan[i] = parameters[i];

        Engine.InvalidatePersistentTensor(_gamma);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        if (_gammaGradient == null)
            return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return _gammaGradient.ToVector();
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _gammaGradient = null;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null)
            throw new InvalidOperationException(
                "Backward pass must be called before updating parameters.");

        // Update in-place to preserve engine persistent-tensor references.
        var updGamma = Engine.TensorSubtract(
            _gamma,
            Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        for (int i = 0; i < _gamma.Length; i++) _gamma[i] = updGamma[i];

        Engine.InvalidatePersistentTensor(_gamma);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _gammaGradient = null;
    }
}

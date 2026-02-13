using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Quantized teacher model with reduced precision for efficient deployment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Quantization reduces the numerical precision of model weights
/// and activations to use fewer bits (e.g., 8-bit instead of 32-bit floating point).
/// This enables:</para>
/// <list type="bullet">
/// <item><description>Smaller model size</description></item>
/// <item><description>Faster inference on hardware with integer support</description></item>
/// <item><description>Reduced memory bandwidth requirements</description></item>
/// </list>
/// <para><b>JIT Support:</b> When constructed with an IJitCompilable base model, this teacher
/// supports JIT compilation using FakeQuantization with Straight-Through Estimator (STE).
/// This allows the quantized model to be differentiated during training while simulating
/// quantization effects.</para>
/// </remarks>
public class QuantizedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>? _baseTeacher;
    private readonly IJitCompilable<T>? _jitCompilableBase;
    private readonly int _quantizationBits;
    private readonly int _outputDim;
    private readonly T _scale;
    private readonly T _zeroPoint;
    private readonly bool _symmetric;

    /// <summary>
    /// Gets the output dimension of the teacher model.
    /// </summary>
    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Initializes a new instance of QuantizedTeacherModel wrapping a teacher interface.
    /// </summary>
    /// <param name="baseTeacher">The base teacher model to quantize.</param>
    /// <param name="quantizationBits">Number of bits for quantization (1-32).</param>
    /// <remarks>
    /// <para>This constructor uses dynamic quantization (per-batch min/max finding) which
    /// does not support JIT compilation. Use the constructor with IJitCompilable for JIT support.</para>
    /// </remarks>
    public QuantizedTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>> baseTeacher,
        int quantizationBits = 8)
    {
        Guard.NotNull(baseTeacher);
        _baseTeacher = baseTeacher;
        _quantizationBits = quantizationBits;
        _outputDim = baseTeacher.OutputDimension;
        _jitCompilableBase = null;
        _scale = NumOps.One;
        _zeroPoint = NumOps.Zero;
        _symmetric = true;

        if (quantizationBits < 1 || quantizationBits > 32)
            throw new ArgumentException("Quantization bits must be between 1 and 32");
    }

    /// <summary>
    /// Initializes a new instance of QuantizedTeacherModel wrapping a JIT-compilable model.
    /// </summary>
    /// <param name="jitCompilableBase">The JIT-compilable base model to quantize.</param>
    /// <param name="outputDimension">Output dimension of the model.</param>
    /// <param name="quantizationBits">Number of bits for quantization (1-32).</param>
    /// <param name="scale">Scale factor for quantization. If default, uses 1/(2^(bits-1)).</param>
    /// <param name="zeroPoint">Zero point for asymmetric quantization. Default is 0.</param>
    /// <param name="symmetric">Whether to use symmetric quantization (centered at 0).</param>
    /// <remarks>
    /// <para><b>JIT Support:</b> This constructor enables JIT compilation using FakeQuantization
    /// with Straight-Through Estimator (STE). The scale and zero point are fixed at construction
    /// time, allowing the graph to be statically compiled.</para>
    /// <para><b>Symmetric vs Asymmetric:</b></para>
    /// <list type="bullet">
    /// <item><description>Symmetric: Range is [-max, max], zero point is 0. Good for weights.</description></item>
    /// <item><description>Asymmetric: Range is [min, max], zero point may be non-zero. Good for activations with bias.</description></item>
    /// </list>
    /// </remarks>
    public QuantizedTeacherModel(
        IJitCompilable<T> jitCompilableBase,
        int outputDimension,
        int quantizationBits = 8,
        T? scale = default,
        T? zeroPoint = default,
        bool symmetric = true)
    {
        Guard.NotNull(jitCompilableBase);
        _jitCompilableBase = jitCompilableBase;
        _quantizationBits = quantizationBits;
        _outputDim = outputDimension;
        _baseTeacher = null;
        _symmetric = symmetric;

        // Default scale: 1/(2^(bits-1)) for symmetric quantization
        if (scale == null || NumOps.Equals(scale, default(T)!))
        {
            double defaultScale = 1.0 / (1 << (quantizationBits - 1));
            _scale = NumOps.FromDouble(defaultScale);
        }
        else
        {
            _scale = scale;
        }

        _zeroPoint = zeroPoint ?? NumOps.Zero;

        if (quantizationBits < 1 || quantizationBits > 32)
            throw new ArgumentException("Quantization bits must be between 1 and 32");
    }

    /// <summary>
    /// Gets quantized logits from the teacher model.
    /// </summary>
    /// <param name="input">Input to the model.</param>
    /// <returns>Quantized logits.</returns>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (_jitCompilableBase != null)
        {
            // IJitCompilable doesn't have execution methods - need to cast to a model interface
            if (_jitCompilableBase is IModel<Vector<T>, Vector<T>, ModelMetadata<T>> model)
            {
                var logits = model.Predict(input);
                return QuantizeFixedScale(logits);
            }

            throw new InvalidOperationException(
                "Underlying model must implement IModel<Vector<T>, Vector<T>, ModelMetadata<T>> to execute predictions. " +
                "IJitCompilable only provides computation graph export for JIT compilation.");
        }

        if (_baseTeacher == null)
            throw new InvalidOperationException("No base teacher or JIT-compilable model configured");

        var baseLogits = _baseTeacher.GetLogits(input);
        return QuantizeDynamic(baseLogits);
    }

    /// <summary>
    /// Applies fixed-scale quantization (JIT-compatible).
    /// </summary>
    private Vector<T> QuantizeFixedScale(Vector<T> vector)
    {
        int n = vector.Length;
        var result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Apply fake quantization: round(x / scale) * scale
            double value = Convert.ToDouble(vector[i]);
            double scaleVal = Convert.ToDouble(_scale);
            double zpVal = Convert.ToDouble(_zeroPoint);

            // Quantize: clamp(round((x - zp) / scale))
            double scaled = (value - zpVal) / scaleVal;
            double quantized = Math.Round(scaled);

            // Clamp to quantization range
            double qmin = _symmetric ? -(1 << (_quantizationBits - 1)) : 0;
            double qmax = _symmetric ? (1 << (_quantizationBits - 1)) - 1 : (1 << _quantizationBits) - 1;
            quantized = Math.Max(qmin, Math.Min(qmax, quantized));

            // Dequantize
            double dequantized = quantized * scaleVal + zpVal;
            result[i] = NumOps.FromDouble(dequantized);
        }

        return result;
    }

    /// <summary>
    /// Applies dynamic quantization (per-batch min/max).
    /// </summary>
    private Vector<T> QuantizeDynamic(Vector<T> vector)
    {
        int n = vector.Length;
        var result = new Vector<T>(n);
        double scale = (1 << _quantizationBits) - 1;

        T minVal = vector[0], maxVal = vector[0];
        for (int i = 1; i < n; i++)
        {
            if (NumOps.LessThan(vector[i], minVal)) minVal = vector[i];
            if (NumOps.GreaterThan(vector[i], maxVal)) maxVal = vector[i];
        }

        double range = Convert.ToDouble(NumOps.Subtract(maxVal, minVal));
        if (range < 1e-10) return vector;

        for (int i = 0; i < n; i++)
        {
            double normalized = (Convert.ToDouble(vector[i]) - Convert.ToDouble(minVal)) / range;
            int quantized = (int)(normalized * scale);
            double dequantized = Convert.ToDouble(minVal) + (quantized / scale) * range;
            result[i] = NumOps.FromDouble(dequantized);
        }

        return result;
    }

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if constructed with an IJitCompilable model that supports JIT;
    /// <c>false</c> if using dynamic quantization with runtime min/max finding.
    /// </value>
    public override bool SupportsJitCompilation => _jitCompilableBase?.SupportsJitCompilation ?? false;

    /// <summary>
    /// Exports the computation graph for JIT compilation with FakeQuantization.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node with quantization applied.</returns>
    /// <exception cref="NotSupportedException">Thrown when using dynamic quantization mode.</exception>
    /// <remarks>
    /// <para>
    /// When constructed with an IJitCompilable model, this method exports the base model's
    /// computation graph and wraps the output with a FakeQuantization operation. The FakeQuantization
    /// uses Straight-Through Estimator (STE) for gradients, allowing backpropagation through
    /// the quantization operation.
    /// </para>
    /// <para>
    /// When using dynamic quantization (per-batch min/max), JIT compilation is not supported
    /// because the quantization parameters are computed at runtime.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_jitCompilableBase != null && _jitCompilableBase.SupportsJitCompilation)
        {
            // Export base model's computation graph
            var baseOutput = _jitCompilableBase.ExportComputationGraph(inputNodes);

            // Apply FakeQuantization to the output
            return TensorOperations<T>.FakeQuantize(
                baseOutput,
                _quantizationBits,
                _scale,
                _zeroPoint,
                _symmetric);
        }

        return ThrowJitNotSupported(
            nameof(QuantizedTeacherModel<T>),
            "it uses dynamic quantization with runtime min/max finding. Use the constructor with an IJitCompilable model for JIT support with fixed-scale quantization");
    }
}

using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class QuantizedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>? _baseTeacher;
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
        _scale = NumOps.One;
        _zeroPoint = NumOps.Zero;
        _symmetric = true;

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
        if (_baseTeacher == null)
            throw new InvalidOperationException("No base teacher configured");

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
}

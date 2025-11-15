using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Quantized teacher model with reduced precision for efficient deployment.
/// </summary>
public class QuantizedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;
    private readonly int _quantizationBits;

    public override int OutputDimension => _baseTeacher.OutputDimension;

    public QuantizedTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>> baseTeacher,
        int quantizationBits = 8)
    {
        _baseTeacher = baseTeacher ?? throw new ArgumentNullException(nameof(baseTeacher));
        _quantizationBits = quantizationBits;
        if (quantizationBits < 1 || quantizationBits > 32)
            throw new ArgumentException("Quantization bits must be between 1 and 32");
    }

    public override Vector<T> GetLogits(Vector<T> input)
    {
        var logits = _baseTeacher.GetLogits(input);
        return Quantize(logits);
    }

    private Vector<T> Quantize(Vector<T> vector)
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

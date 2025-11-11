using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Transformer-based teacher model with attention mechanism support.
/// </summary>
public class TransformerTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>> _forwardFunc;
    private readonly Func<Vector<T>, string, object?>? _attentionExtractor;
    private readonly int _outputDim;

    public override int OutputDimension => _outputDim;

    public TransformerTeacherModel(
        Func<Vector<T>, Vector<T>> forwardFunc,
        int outputDimension,
        Func<Vector<T>, string, object?>? attentionExtractor = null)
    {
        _forwardFunc = forwardFunc ?? throw new ArgumentNullException(nameof(forwardFunc));
        _outputDim = outputDimension;
        _attentionExtractor = attentionExtractor;
    }

    public override Vector<T> GetLogits(Vector<T> input) => _forwardFunc(input);

    public override object? GetAttentionWeights(Vector<T> input, string layerName) =>
        _attentionExtractor?.Invoke(input, layerName);

    protected override Vector<T> ApplyTemperatureSoftmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }
}

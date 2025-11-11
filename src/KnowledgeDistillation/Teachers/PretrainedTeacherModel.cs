using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Pretrained teacher model from external source (e.g., ImageNet, BERT).
/// </summary>
public class PretrainedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>> _pretrainedForward;
    private readonly int _outputDim;

    public override int OutputDimension => _outputDim;

    public PretrainedTeacherModel(
        Func<Vector<T>, Vector<T>> pretrainedForward,
        int outputDimension)
    {
        _pretrainedForward = pretrainedForward ?? throw new ArgumentNullException(nameof(pretrainedForward));
        _outputDim = outputDimension;
    }

    public override Vector<T> GetLogits(Vector<T> input) => _pretrainedForward(input);

    protected override Vector<T> ApplyTemperatureSoftmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }
}

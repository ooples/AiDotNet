using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Multi-modal teacher that combines multiple input modalities (vision, text, audio).
/// </summary>
public class MultiModalTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _modalityTeachers;
    private readonly double[] _modalityWeights;

    public override int OutputDimension => _modalityTeachers[0].OutputDimension;

    public MultiModalTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] modalityTeachers,
        double[]? modalityWeights = null)
    {
        _modalityTeachers = modalityTeachers ?? throw new ArgumentNullException(nameof(modalityTeachers));

        if (modalityWeights == null)
        {
            _modalityWeights = Enumerable.Repeat(1.0 / modalityTeachers.Length, modalityTeachers.Length).ToArray();
        }
        else
        {
            if (modalityWeights.Length != modalityTeachers.Length)
                throw new ArgumentException("Modality weights must match number of teachers");
            _modalityWeights = modalityWeights;
        }
    }

    public override Vector<T> GetLogits(Vector<T> input)
    {
        int n = _modalityTeachers[0].OutputDimension;
        var combined = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < _modalityTeachers.Length; i++)
            {
                var logits = _modalityTeachers[i].GetLogits(input);
                var weighted = NumOps.Multiply(logits[j], NumOps.FromDouble(_modalityWeights[i]));
                sum = NumOps.Add(sum, weighted);
            }
            combined[j] = sum;
        }

        return combined;
    }

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

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Curriculum teacher that gradually increases task difficulty during training.
/// </summary>
public class CurriculumTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;
    private readonly CurriculumStrategy _strategy;
    private double _currentDifficulty;

    public override int OutputDimension => _baseTeacher.OutputDimension;

    public CurriculumTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>> baseTeacher,
        CurriculumStrategy strategy = CurriculumStrategy.EasyToHard)
    {
        _baseTeacher = baseTeacher ?? throw new ArgumentNullException(nameof(baseTeacher));
        _strategy = strategy;
        _currentDifficulty = 0.0;
    }

    public void UpdateDifficulty(double difficulty) => _currentDifficulty = Math.Clamp(difficulty, 0.0, 1.0);

    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);

    protected override Vector<T> ApplyTemperatureSoftmax(Vector<T> logits, double temperature)
    {
        double adjustedTemp = _strategy == CurriculumStrategy.EasyToHard
            ? temperature * (1.0 + _currentDifficulty)
            : temperature * (2.0 - _currentDifficulty);

        return Softmax(logits, adjustedTemp);
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
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

public enum CurriculumStrategy
{
    EasyToHard,
    HardToEasy
}

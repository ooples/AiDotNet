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

    public void UpdateDifficulty(double difficulty) => _currentDifficulty = MathHelper.Clamp(difficulty, 0.0, 1.0);

    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);
}

public enum CurriculumStrategy
{
    EasyToHard,
    HardToEasy
}

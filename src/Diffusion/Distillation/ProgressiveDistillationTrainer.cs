using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for progressive distillation that halves the number of steps in each round.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Progressive distillation trains a student to match the output of a teacher's two steps
/// in a single step. Applied iteratively: 1024→512→256→...→4→2→1 steps. Each round halves
/// the step count while maintaining quality, producing a model that generates in few steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine learning to draw in fewer strokes. First, you learn to
/// do in 512 strokes what took 1024. Then 256, then 128, and so on. Each time you learn
/// to combine two teacher steps into one student step. After several rounds, you can
/// draw the same picture in just 4-8 strokes.
/// </para>
/// <para>
/// Reference: Salimans and Ho, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022
/// </para>
/// </remarks>
public class ProgressiveDistillationTrainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly int _initialSteps;
    private readonly int _targetSteps;

    /// <summary>
    /// Initializes a new progressive distillation trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher model (at current step count).</param>
    /// <param name="initialSteps">Teacher's current step count (default: 1024).</param>
    /// <param name="targetSteps">Target step count after all rounds (default: 4).</param>
    public ProgressiveDistillationTrainer(
        IDiffusionModel<T> teacher,
        int initialSteps = 1024,
        int targetSteps = 4)
    {
        _teacher = teacher;
        _initialSteps = initialSteps;
        _targetSteps = targetSteps;
    }

    /// <summary>
    /// Computes the progressive distillation loss (student one-step vs teacher two-step).
    /// </summary>
    /// <param name="studentOneStep">Student's single-step output from timestep t to t-2h.</param>
    /// <param name="teacherTwoStep">Teacher's two-step output from t→t-h→t-2h.</param>
    /// <returns>L2 distillation loss.</returns>
    public T ComputeDistillationLoss(Vector<T> studentOneStep, Vector<T> teacherTwoStep)
    {
        var loss = NumOps.Zero;
        for (int i = 0; i < studentOneStep.Length && i < teacherTwoStep.Length; i++)
        {
            var diff = NumOps.Subtract(studentOneStep[i], teacherTwoStep[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }
        return NumOps.Divide(loss, NumOps.FromDouble(studentOneStep.Length));
    }

    /// <summary>
    /// Gets the number of distillation rounds needed.
    /// </summary>
    public int NumRounds => (int)Math.Ceiling(Math.Log(_initialSteps / (double)_targetSteps, 2));

    /// <summary>
    /// Gets the teacher model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;

    /// <summary>
    /// Gets the initial step count.
    /// </summary>
    public int InitialSteps => _initialSteps;

    /// <summary>
    /// Gets the target step count.
    /// </summary>
    public int TargetSteps => _targetSteps;
}

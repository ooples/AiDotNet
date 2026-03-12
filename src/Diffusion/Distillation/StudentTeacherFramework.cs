using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Generic student-teacher framework for knowledge distillation in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Provides the foundational abstraction for all distillation approaches where a smaller/faster
/// student model learns to replicate the behavior of a larger/slower teacher model. Handles
/// EMA (Exponential Moving Average) target network updates, timestep sampling strategies,
/// and loss computation delegation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like a tutoring framework. A knowledgeable teacher (the full
/// diffusion model) teaches a student (a faster model) to produce similar results. The framework
/// manages how the teacher shows examples and how the student learns, supporting different
/// teaching styles (consistency, progressive, adversarial distillation).
/// </para>
/// <para>
/// Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS Workshop 2015;
/// adapted for diffusion models by Salimans &amp; Ho, "Progressive Distillation for Fast Sampling
/// of Diffusion Models", ICLR 2022
/// </para>
/// </remarks>
public class StudentTeacherFramework<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly IDiffusionModel<T> _student;
    private readonly double _emaDecay;
    private readonly double _temperatureScale;
    private Vector<T>? _emaParameters;

    /// <summary>
    /// Gets the teacher diffusion model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;

    /// <summary>
    /// Gets the student diffusion model being trained.
    /// </summary>
    public IDiffusionModel<T> Student => _student;

    /// <summary>
    /// Gets the EMA decay rate for the target network.
    /// </summary>
    public double EMADecay => _emaDecay;

    /// <summary>
    /// Initializes a new student-teacher framework.
    /// </summary>
    /// <param name="teacher">Pretrained teacher model (frozen during distillation).</param>
    /// <param name="student">Student model to be trained.</param>
    /// <param name="emaDecay">EMA decay rate for the target network (default: 0.9999).</param>
    /// <param name="temperatureScale">Temperature for softening teacher outputs (default: 1.0).</param>
    public StudentTeacherFramework(
        IDiffusionModel<T> teacher,
        IDiffusionModel<T> student,
        double emaDecay = 0.9999,
        double temperatureScale = 1.0)
    {
        _teacher = teacher;
        _student = student;
        _emaDecay = emaDecay;
        _temperatureScale = temperatureScale;
    }

    /// <summary>
    /// Computes the distillation loss between student and teacher outputs.
    /// </summary>
    /// <param name="studentOutput">Student model's prediction.</param>
    /// <param name="teacherOutput">Teacher model's prediction (detached/frozen).</param>
    /// <returns>MSE distillation loss.</returns>
    public T ComputeDistillationLoss(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        var loss = NumOps.Zero;
        var temp = NumOps.FromDouble(_temperatureScale);
        int len = Math.Min(studentOutput.Length, teacherOutput.Length);

        for (int i = 0; i < len; i++)
        {
            var scaledStudent = NumOps.Divide(studentOutput[i], temp);
            var scaledTeacher = NumOps.Divide(teacherOutput[i], temp);
            var diff = NumOps.Subtract(scaledStudent, scaledTeacher);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(len));
    }

    /// <summary>
    /// Updates the EMA target network parameters from the student's current parameters.
    /// </summary>
    /// <param name="studentParams">Current student model parameters.</param>
    /// <returns>Updated EMA parameters.</returns>
    public Vector<T> UpdateEMA(Vector<T> studentParams)
    {
        if (_emaParameters == null)
        {
            _emaParameters = new Vector<T>(studentParams.Length);
            for (int i = 0; i < studentParams.Length; i++)
                _emaParameters[i] = studentParams[i];
            return _emaParameters;
        }

        var decay = NumOps.FromDouble(_emaDecay);
        var oneMinusDecay = NumOps.Subtract(NumOps.One, decay);

        for (int i = 0; i < _emaParameters.Length && i < studentParams.Length; i++)
        {
            _emaParameters[i] = NumOps.Add(
                NumOps.Multiply(decay, _emaParameters[i]),
                NumOps.Multiply(oneMinusDecay, studentParams[i]));
        }

        return _emaParameters;
    }

    /// <summary>
    /// Samples a random timestep for training.
    /// </summary>
    /// <param name="numTimesteps">Total number of timesteps in the schedule.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>Sampled timestep value.</returns>
    public T SampleTimestep(int numTimesteps, Random random)
    {
        var t = random.NextDouble();
        return NumOps.FromDouble(t * numTimesteps);
    }
}

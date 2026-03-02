using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for Trajectory Consistency Distillation (TCD) with trajectory-aware loss.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TCD extends standard consistency distillation by considering the entire ODE trajectory
/// rather than individual timestep pairs. The trajectory-aware loss ensures the student
/// is self-consistent across the entire denoising path, producing higher quality at
/// any step count.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard consistency distillation checks if the student gets
/// the right answer at each individual step. TCD also checks if the student's answers
/// are consistent with each other across the whole journey — like checking that a
/// student's essay tells a coherent story, not just that each sentence is correct.
/// </para>
/// <para>
/// Reference: Zheng et al., "Trajectory Consistency Distillation", 2024
/// </para>
/// </remarks>
public class TrajectoryConsistencyDistiller<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly double _emaDecay;
    private readonly double _stochasticEta;

    /// <summary>
    /// Initializes a new TCD trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher model.</param>
    /// <param name="emaDecay">EMA decay for target network (default: 0.999).</param>
    /// <param name="stochasticEta">Stochastic noise injection strength during inference (default: 0.3).</param>
    public TrajectoryConsistencyDistiller(
        IDiffusionModel<T> teacher,
        double emaDecay = 0.999,
        double stochasticEta = 0.3)
    {
        _teacher = teacher;
        _emaDecay = emaDecay;
        _stochasticEta = stochasticEta;
    }

    /// <summary>
    /// Computes the trajectory consistency loss across multiple timestep pairs.
    /// </summary>
    /// <param name="studentOutputs">Student predictions at trajectory timesteps.</param>
    /// <param name="targetOutputs">EMA target predictions at corresponding timesteps.</param>
    /// <returns>Averaged trajectory consistency loss.</returns>
    public T ComputeTrajectoryLoss(Vector<T>[] studentOutputs, Vector<T>[] targetOutputs)
    {
        var totalLoss = NumOps.Zero;
        int pairs = Math.Min(studentOutputs.Length, targetOutputs.Length);

        for (int p = 0; p < pairs; p++)
        {
            var student = studentOutputs[p];
            var target = targetOutputs[p];
            var pairLoss = NumOps.Zero;

            for (int i = 0; i < student.Length && i < target.Length; i++)
            {
                var diff = NumOps.Subtract(student[i], target[i]);
                pairLoss = NumOps.Add(pairLoss, NumOps.Multiply(diff, diff));
            }

            totalLoss = NumOps.Add(totalLoss, NumOps.Divide(pairLoss, NumOps.FromDouble(student.Length)));
        }

        return pairs > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(pairs)) : NumOps.Zero;
    }

    /// <summary>
    /// Gets the teacher model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;

    /// <summary>
    /// Gets the EMA decay rate.
    /// </summary>
    public double EMADecay => _emaDecay;

    /// <summary>
    /// Gets the stochastic noise injection strength.
    /// </summary>
    public double StochasticEta => _stochasticEta;
}

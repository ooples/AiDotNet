using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for consistency distillation from a pretrained diffusion model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the consistency distillation procedure where a student model learns to map
/// any point on the ODE trajectory directly to the trajectory's origin (clean data).
/// The teacher provides target outputs via one-step ODE evaluation, and the student
/// learns to match these targets with a consistency loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> Given a pretrained diffusion model (teacher), this trainer
/// creates a fast student model. The teacher shows the student what the clean image
/// should look like from any noise level, and the student learns to jump directly
/// to that clean image in one step.
/// </para>
/// <para>
/// Reference: Song et al., "Consistency Models", ICML 2023
/// </para>
/// </remarks>
public class ConsistencyDistillationTrainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _teacher;
    private readonly double _emaDecay;
    private readonly int _numTimestepBins;

    /// <summary>
    /// Initializes a new consistency distillation trainer.
    /// </summary>
    /// <param name="teacher">Pretrained teacher diffusion model.</param>
    /// <param name="emaDecay">EMA decay rate for target network (default: 0.999).</param>
    /// <param name="numTimestepBins">Number of discretization bins (default: 120).</param>
    public ConsistencyDistillationTrainer(
        IDiffusionModel<T> teacher,
        double emaDecay = 0.999,
        int numTimestepBins = 120)
    {
        _teacher = teacher;
        _emaDecay = emaDecay;
        _numTimestepBins = numTimestepBins;
    }

    /// <summary>
    /// Computes the consistency distillation loss for a training batch.
    /// </summary>
    /// <param name="studentOutput">Student model prediction at timestep t_n.</param>
    /// <param name="targetOutput">Target (EMA model) prediction at timestep t_{n+1}.</param>
    /// <returns>Pseudo-Huber consistency loss value.</returns>
    public T ComputeConsistencyLoss(Vector<T> studentOutput, Vector<T> targetOutput)
    {
        // Pseudo-Huber loss: sqrt((student - target)^2 + c^2) - c
        var c = NumOps.FromDouble(0.00054 * Math.Sqrt(studentOutput.Length));
        var cSquared = NumOps.Multiply(c, c);
        var loss = NumOps.Zero;

        for (int i = 0; i < studentOutput.Length && i < targetOutput.Length; i++)
        {
            var diff = NumOps.Subtract(studentOutput[i], targetOutput[i]);
            var diffSquared = NumOps.Multiply(diff, diff);
            var inner = NumOps.Add(diffSquared, cSquared);
            var sqrtInner = NumOps.Sqrt(inner);
            loss = NumOps.Add(loss, NumOps.Subtract(sqrtInner, c));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(studentOutput.Length));
    }

    /// <summary>
    /// Gets the EMA decay rate.
    /// </summary>
    public double EMADecay => _emaDecay;

    /// <summary>
    /// Gets the number of timestep discretization bins.
    /// </summary>
    public int NumTimestepBins => _numTimestepBins;

    /// <summary>
    /// Gets the teacher model.
    /// </summary>
    public IDiffusionModel<T> Teacher => _teacher;
}

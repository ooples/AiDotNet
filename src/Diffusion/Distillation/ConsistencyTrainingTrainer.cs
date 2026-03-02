using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Distillation;

/// <summary>
/// Trainer for consistency training from scratch without a pretrained teacher model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike consistency distillation (which requires a pretrained teacher), consistency training
/// learns the consistency function directly from data using a self-consistency loss. The model
/// learns that mapping any point along the same ODE trajectory should produce the same output,
/// without needing teacher guidance. Uses a progressive schedule that increases discretization
/// steps during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> While consistency distillation needs a "teacher" model to learn from,
/// consistency training learns on its own. It discovers that different amounts of noise added
/// to the same image should all map back to the same clean image. This is like learning to
/// recognize a face from photos taken at different exposure levels — they're all the same face.
/// </para>
/// <para>
/// Reference: Song et al., "Improved Techniques for Training Consistency Models", ICML 2024 (iCT);
/// Song and Dhariwal, "Consistency Models", ICML 2023
/// </para>
/// </remarks>
public class ConsistencyTrainingTrainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _emaDecay;
    private readonly int _initialDiscretizationSteps;
    private readonly int _finalDiscretizationSteps;
    private readonly double _sigmaMin;
    private readonly double _sigmaMax;
    private int _currentStep;

    /// <summary>
    /// Gets the initial discretization step count.
    /// </summary>
    public int InitialDiscretizationSteps => _initialDiscretizationSteps;

    /// <summary>
    /// Gets the current training step.
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Initializes a new consistency training trainer.
    /// </summary>
    /// <param name="emaDecay">EMA decay rate for the target network (default: 0.9993).</param>
    /// <param name="initialDiscretizationSteps">Starting number of discretization bins (default: 2).</param>
    /// <param name="finalDiscretizationSteps">Final number of discretization bins (default: 150).</param>
    /// <param name="sigmaMin">Minimum noise level (default: 0.002).</param>
    /// <param name="sigmaMax">Maximum noise level (default: 80.0).</param>
    public ConsistencyTrainingTrainer(
        double emaDecay = 0.9993,
        int initialDiscretizationSteps = 2,
        int finalDiscretizationSteps = 150,
        double sigmaMin = 0.002,
        double sigmaMax = 80.0)
    {
        _emaDecay = emaDecay;
        _initialDiscretizationSteps = initialDiscretizationSteps;
        _finalDiscretizationSteps = finalDiscretizationSteps;
        _sigmaMin = sigmaMin;
        _sigmaMax = sigmaMax;
        _currentStep = 0;
    }

    /// <summary>
    /// Computes the self-consistency loss for a training batch.
    /// </summary>
    /// <param name="modelOutputAtT">Model prediction at timestep t_n (closer to data).</param>
    /// <param name="targetOutputAtTPlusOne">Target (EMA) prediction at timestep t_{n+1} (more noisy).</param>
    /// <returns>Pseudo-Huber consistency loss.</returns>
    public T ComputeConsistencyLoss(Vector<T> modelOutputAtT, Vector<T> targetOutputAtTPlusOne)
    {
        var c = NumOps.FromDouble(0.00054 * Math.Sqrt(modelOutputAtT.Length));
        var cSquared = NumOps.Multiply(c, c);
        var loss = NumOps.Zero;
        int len = Math.Min(modelOutputAtT.Length, targetOutputAtTPlusOne.Length);

        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(modelOutputAtT[i], targetOutputAtTPlusOne[i]);
            var diffSquared = NumOps.Multiply(diff, diff);
            var inner = NumOps.Add(diffSquared, cSquared);
            loss = NumOps.Add(loss, NumOps.Subtract(NumOps.Sqrt(inner), c));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(len));
    }

    /// <summary>
    /// Gets the current number of discretization steps based on training progress.
    /// </summary>
    /// <param name="totalTrainingSteps">Total planned training steps.</param>
    /// <returns>Current discretization step count (increases over training).</returns>
    public int GetCurrentDiscretizationSteps(int totalTrainingSteps)
    {
        if (totalTrainingSteps <= 0) return _initialDiscretizationSteps;
        double progress = Math.Min(1.0, (double)_currentStep / totalTrainingSteps);
        return (int)(_initialDiscretizationSteps +
            progress * (_finalDiscretizationSteps - _initialDiscretizationSteps));
    }

    /// <summary>
    /// Computes the adaptive EMA decay rate based on current discretization.
    /// </summary>
    /// <param name="totalTrainingSteps">Total planned training steps.</param>
    /// <returns>Adapted EMA decay rate.</returns>
    public double GetAdaptiveEMADecay(int totalTrainingSteps)
    {
        int N = GetCurrentDiscretizationSteps(totalTrainingSteps);
        return Math.Exp(Math.Log(_emaDecay) * _initialDiscretizationSteps / N);
    }

    /// <summary>
    /// Advances the training step counter.
    /// </summary>
    public void Step()
    {
        _currentStep++;
    }

    /// <summary>
    /// Gets the sigma (noise level) for a given discretization index.
    /// </summary>
    /// <param name="index">Discretization index.</param>
    /// <param name="numSteps">Total discretization steps.</param>
    /// <returns>Noise level sigma.</returns>
    public T GetSigma(int index, int numSteps)
    {
        if (numSteps <= 1) return NumOps.FromDouble(_sigmaMin);
        double rho = 7.0;
        double fraction = (double)index / (numSteps - 1);
        double sigma = Math.Pow(
            Math.Pow(_sigmaMin, 1.0 / rho) + fraction * (Math.Pow(_sigmaMax, 1.0 / rho) - Math.Pow(_sigmaMin, 1.0 / rho)),
            rho);
        return NumOps.FromDouble(sigma);
    }
}

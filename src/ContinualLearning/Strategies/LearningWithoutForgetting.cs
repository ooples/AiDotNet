using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Learning without Forgetting (LwF) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> LwF prevents forgetting by using knowledge distillation.
/// It works by:
/// 1. Before learning a new task, save the model's current predictions (the "teacher" model)
/// 2. When training on the new task, also try to match the old predictions
/// 3. This preserves the model's knowledge about previous tasks
/// </para>
///
/// <para><b>How it works:</b>
/// - The loss function has two parts:
///   1. Task loss: how well the model predicts the new task
///   2. Distillation loss: how well the model matches its old predictions
/// - The distillation loss ensures the model doesn't forget what it learned before
/// - This is especially useful when you don't have access to old task data
/// </para>
///
/// <para><b>Advantages:</b>
/// - No need to store old task data
/// - Works well when task outputs are similar (e.g., all classification tasks)
/// - Computationally efficient
/// </para>
///
/// <para><b>Reference:</b> Li and Hoiem "Learning without Forgetting" (2017)</para>
/// </remarks>
public class LearningWithoutForgetting<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILossFunction<T> _lossFunction;
    private readonly T _distillationTemperature;
    private readonly T _distillationWeight;

    // Store the teacher model's parameters from previous tasks
    private IFullModel<T, TInput, TOutput>? _teacherModel;

    /// <summary>
    /// Initializes a new LwF strategy.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="distillationTemperature">Temperature for softening probability distributions (higher = softer). Default is 2.0.</param>
    /// <param name="distillationWeight">Weight for the distillation loss relative to task loss. Default is 1.0.</param>
    public LearningWithoutForgetting(
        ILossFunction<T> lossFunction,
        double distillationTemperature = 2.0,
        double distillationWeight = 1.0)
    {
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _distillationTemperature = NumOps.FromDouble(distillationTemperature);
        _distillationWeight = NumOps.FromDouble(distillationWeight);
    }

    /// <inheritdoc/>
    public void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // Clone the current model to serve as the teacher
        if (_teacherModel == null && model.ParameterCount > 0)
        {
            _teacherModel = model.Clone();
        }
        else if (_teacherModel != null)
        {
            // Update teacher with current model parameters
            var currentParams = model.GetParameters();
            _teacherModel.SetParameters(currentParams);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>LwF-Specific Behavior:</b> Unlike EWC which adds a regularization term based on parameter importance,
    /// LwF computes distillation loss during training by comparing teacher and student outputs on actual data.</para>
    ///
    /// <para>The distillation loss is computed using <see cref="ComputeDistillationLoss(Vector{T}, Vector{T})"/>
    /// when training batches are processed. This method returns zero because LwF's regularization happens
    /// through the training loss function, not as a separate parameter-based regularization term.</para>
    ///
    /// <para>To use LwF properly:
    /// 1. Call this method to get the parameter regularization (always zero for LwF)
    /// 2. During training, call ComputeDistillationLoss(teacherOutput, studentOutput) for each batch
    /// 3. Add both the task loss and distillation loss together: L_total = L_task + λ * L_distill
    /// </para>
    /// </remarks>
    public T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // LwF does not use parameter-based regularization like EWC.
        // Instead, it uses knowledge distillation during training, which requires
        // actual data to pass through both teacher and student models.
        // Use ComputeDistillationLoss(teacherOutput, studentOutput) during training.
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // LwF adjusts the loss function, not the gradients directly
        // The gradient adjustment happens through backpropagation of the distillation loss
        return gradients;
    }

    /// <inheritdoc/>
    public void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Update the teacher model with the newly trained model
        var currentParams = model.GetParameters();

        if (_teacherModel == null)
        {
            _teacherModel = model.Clone();
        }
        else
        {
            _teacherModel.SetParameters(currentParams);
        }
    }

    /// <summary>
    /// Computes the distillation loss between teacher and student predictions.
    /// </summary>
    /// <param name="teacherOutput">Output from the teacher model.</param>
    /// <param name="studentOutput">Output from the student model.</param>
    /// <returns>The distillation loss value.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> The distillation loss is typically computed as:
    /// L_distill = T^2 * KL(softmax(teacher / T) || softmax(student / T))
    /// where T is the temperature parameter.
    /// </para>
    ///
    /// <para>The temperature T softens the probability distributions:
    /// - T = 1: normal softmax
    /// - T &gt; 1: softer distributions (emphasizes smaller probabilities)
    /// - Higher T makes the student learn more from the teacher's uncertainty
    /// </para>
    /// </remarks>
    public T ComputeDistillationLoss(Vector<T> teacherOutput, Vector<T> studentOutput)
    {
        if (teacherOutput.Length != studentOutput.Length)
            throw new ArgumentException("Teacher and student outputs must have the same length");

        // Apply temperature scaling to both outputs
        var softTeacher = SoftmaxWithTemperature(teacherOutput, _distillationTemperature);
        var softStudent = SoftmaxWithTemperature(studentOutput, _distillationTemperature);

        // Compute KL divergence: sum(teacher * log(teacher / student))
        T kl = NumOps.Zero;
        for (int i = 0; i < softTeacher.Length; i++)
        {
            if (Convert.ToDouble(softTeacher[i]) > 1e-10) // Avoid log(0)
            {
                var ratio = NumOps.Divide(softTeacher[i],
                    NumOps.Add(softStudent[i], NumOps.FromDouble(1e-10)));
                var logRatio = NumOps.FromDouble(Math.Log(Convert.ToDouble(ratio)));
                var term = NumOps.Multiply(softTeacher[i], logRatio);
                kl = NumOps.Add(kl, term);
            }
        }

        // Scale by T^2 and distillation weight
        var tempSquared = NumOps.Multiply(_distillationTemperature, _distillationTemperature);
        kl = NumOps.Multiply(kl, tempSquared);
        kl = NumOps.Multiply(kl, _distillationWeight);

        return kl;
    }

    /// <summary>
    /// Applies softmax with temperature scaling.
    /// </summary>
    private Vector<T> SoftmaxWithTemperature(Vector<T> logits, T temperature)
    {
        var scaled = new T[logits.Length];
        T maxLogit = logits[0];

        // Find max for numerical stability
        for (int i = 1; i < logits.Length; i++)
        {
            if (Convert.ToDouble(logits[i]) > Convert.ToDouble(maxLogit))
                maxLogit = logits[i];
        }

        // Compute exp((logit - max) / T) and sum
        T sum = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            var divided = NumOps.Divide(shifted, temperature);
            var expValue = NumOps.FromDouble(Math.Exp(Convert.ToDouble(divided)));
            scaled[i] = expValue;
            sum = NumOps.Add(sum, expValue);
        }

        // Normalize
        for (int i = 0; i < scaled.Length; i++)
        {
            scaled[i] = NumOps.Divide(scaled[i], sum);
        }

        return new Vector<T>(scaled);
    }

    /// <summary>
    /// Gets the teacher model used for knowledge distillation.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? TeacherModel => _teacherModel;
}

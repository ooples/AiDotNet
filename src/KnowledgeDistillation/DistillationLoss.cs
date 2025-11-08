using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Implements the standard knowledge distillation loss function (Hinton et al., 2015).
/// Combines hard loss (true labels) with soft loss (teacher predictions) using temperature scaling.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class calculates how well a student model is learning from
/// a teacher model. It uses two types of information:</para>
///
/// <para>1. **Soft Targets** from the teacher: The teacher's predictions, softened with temperature,
/// reveal relationships between classes (e.g., "dog" is more similar to "cat" than to "car").</para>
///
/// <para>2. **Hard Targets** (true labels): The actual correct answers for the training data.</para>
///
/// <para>The mathematics behind distillation loss:</para>
/// <code>
/// L_soft = KL_Divergence(softmax(student/T), softmax(teacher/T)) × T²
/// L_hard = CrossEntropy(softmax(student), true_labels)
/// L_total = α × L_hard + (1 - α) × L_soft
///
/// Where:
///   T = temperature (typically 2-10)
///   α = alpha, balance between hard and soft loss (typically 0.3-0.5)
///   T² scaling balances gradient magnitudes
/// </code>
///
/// <para><b>Real-world Example:</b>
/// Imagine teaching a student to recognize animals. Instead of just saying "this is a dog" (hard label),
/// you explain "this is definitely a dog (90% confident), it could be mistaken for a wolf (8%),
/// but definitely not a cat (1%) or car (1%)". This richer information helps the student learn
/// better relationships between concepts.</para>
///
/// <para><b>References:</b>
/// - Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531</para>
/// </remarks>
public class DistillationLoss<T> : IDistillationStrategy<Vector<T>, T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets or sets the temperature parameter for softening probability distributions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher temperature makes the probability distribution "softer",
    /// revealing more information about the model's uncertainty and class relationships.</para>
    ///
    /// <para>Example with logits [10, 2, 1]:
    /// - T=1: probabilities [0.9999, 0.0001, 0.0000] (very peaked)
    /// - T=3: probabilities [0.952, 0.024, 0.024] (softer, more information)
    /// - T=5: probabilities [0.866, 0.067, 0.067] (even softer)</para>
    ///
    /// <para>Typical values: 3-5 for image classification, 2-4 for language models.</para>
    /// </remarks>
    public double Temperature { get; set; }

    /// <summary>
    /// Gets or sets the balance parameter between hard loss and soft loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much the student learns from true labels vs. teacher:
    /// - α = 0: Only learn from teacher (useful when labels are noisy)
    /// - α = 0.3: 30% from labels, 70% from teacher (common default)
    /// - α = 0.5: Equal weight to both sources
    /// - α = 1: Only learn from labels (no distillation)</para>
    /// </remarks>
    public double Alpha { get; set; }

    /// <summary>
    /// Initializes a new instance of the DistillationLoss class.
    /// </summary>
    /// <param name="temperature">Softmax temperature for distillation (default: 3.0). Higher values (2-10)
    /// produce softer probability distributions that transfer more knowledge.</param>
    /// <param name="alpha">Balance between hard loss and soft loss (default: 0.3). Lower values give
    /// more weight to the teacher's knowledge.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default values (temperature=3.0, alpha=0.3) work well for most
    /// classification tasks. You may want to adjust them based on your specific problem:
    /// - Increase temperature if the teacher's uncertainty is important (complex tasks)
    /// - Decrease alpha if you have noisy labels or want to rely more on the teacher
    /// - Increase alpha if you have very clean labels and a smaller capacity gap</para>
    /// </remarks>
    public DistillationLoss(double temperature = 3.0, double alpha = 0.3)
    {
        if (temperature <= 0)
            throw new ArgumentException("Temperature must be positive", nameof(temperature));
        if (alpha < 0 || alpha > 1)
            throw new ArgumentException("Alpha must be between 0 and 1", nameof(alpha));

        _numOps = MathHelper.GetNumericOperations<T>();
        Temperature = temperature;
        Alpha = alpha;
    }

    /// <summary>
    /// Computes the combined distillation loss (soft loss from teacher + hard loss from true labels).
    /// </summary>
    /// <param name="studentLogits">The student model's raw outputs (logits) before softmax.</param>
    /// <param name="teacherLogits">The teacher model's raw outputs (logits) before softmax.</param>
    /// <param name="trueLabels">Ground truth labels as one-hot vectors (optional). If null, only soft loss is computed.</param>
    /// <returns>The total distillation loss combining soft and hard components.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main loss function that guides student training.
    /// It combines two objectives:
    /// 1. Match the teacher's soft predictions (learn the teacher's knowledge)
    /// 2. Match the true labels (learn to be correct)</para>
    ///
    /// <para>The soft loss uses KL divergence, which measures how different two probability
    /// distributions are. When the student's soft predictions match the teacher's, KL divergence
    /// approaches zero.</para>
    /// </remarks>
    public T ComputeLoss(Vector<T> studentLogits, Vector<T> teacherLogits, Vector<T>? trueLabels = null)
    {
        if (studentLogits.Length != teacherLogits.Length)
            throw new ArgumentException("Student and teacher logits must have the same length");
        if (trueLabels != null && studentLogits.Length != trueLabels.Length)
            throw new ArgumentException("Logits and labels must have the same length");

        // Compute soft loss: KL divergence between temperature-scaled distributions
        var studentSoft = Softmax(studentLogits, Temperature);
        var teacherSoft = Softmax(teacherLogits, Temperature);

        var softLoss = KLDivergence(studentSoft, teacherSoft);

        // Scale by T² to balance gradient magnitudes
        // This is crucial: without T² scaling, the soft loss gradients would be too small
        softLoss = _numOps.Multiply(softLoss, _numOps.FromDouble(Temperature * Temperature));

        // If we have true labels, add hard loss
        if (trueLabels != null)
        {
            var studentProbs = Softmax(studentLogits, temperature: 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);

            // Combine: α × hard_loss + (1 - α) × soft_loss
            var alphaT = _numOps.FromDouble(Alpha);
            var oneMinusAlpha = _numOps.FromDouble(1.0 - Alpha);

            var totalLoss = _numOps.Add(
                _numOps.Multiply(alphaT, hardLoss),
                _numOps.Multiply(oneMinusAlpha, softLoss)
            );

            return totalLoss;
        }

        return softLoss;
    }

    /// <summary>
    /// Computes the gradient of the distillation loss for backpropagation.
    /// </summary>
    /// <param name="studentLogits">The student model's raw outputs (logits).</param>
    /// <param name="teacherLogits">The teacher model's raw outputs (logits).</param>
    /// <param name="trueLabels">Ground truth labels (optional).</param>
    /// <returns>Gradient vector with respect to student logits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient tells us how to adjust the student's parameters
    /// to reduce the loss. It points in the direction that increases loss, so we subtract it
    /// (gradient descent) to improve the model.</para>
    ///
    /// <para>The soft gradient: (student_soft - teacher_soft) × T²
    /// The hard gradient: (student_probs - true_labels)
    /// Combined gradient: α × hard_grad + (1 - α) × soft_grad</para>
    /// </remarks>
    public Vector<T> ComputeGradient(Vector<T> studentLogits, Vector<T> teacherLogits, Vector<T>? trueLabels = null)
    {
        if (studentLogits.Length != teacherLogits.Length)
            throw new ArgumentException("Student and teacher logits must have the same length");
        if (trueLabels != null && studentLogits.Length != trueLabels.Length)
            throw new ArgumentException("Logits and labels must have the same length");

        int n = studentLogits.Length;
        var gradient = new Vector<T>(n);

        // Soft gradient: ∂L_soft/∂logits = (student_soft - teacher_soft) × T²
        var studentSoft = Softmax(studentLogits, Temperature);
        var teacherSoft = Softmax(teacherLogits, Temperature);

        for (int i = 0; i < n; i++)
        {
            var diff = _numOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = _numOps.Multiply(diff, _numOps.FromDouble(Temperature * Temperature));
        }

        // If we have true labels, add hard gradient
        if (trueLabels != null)
        {
            var studentProbs = Softmax(studentLogits, temperature: 1.0);
            var hardGradient = new Vector<T>(n);

            // Hard gradient: ∂L_hard/∂logits = student_probs - true_labels
            for (int i = 0; i < n; i++)
            {
                hardGradient[i] = _numOps.Subtract(studentProbs[i], trueLabels[i]);
            }

            // Combine gradients: α × hard_grad + (1 - α) × soft_grad
            var alphaT = _numOps.FromDouble(Alpha);
            var oneMinusAlpha = _numOps.FromDouble(1.0 - Alpha);

            for (int i = 0; i < n; i++)
            {
                gradient[i] = _numOps.Add(
                    _numOps.Multiply(alphaT, hardGradient[i]),
                    _numOps.Multiply(oneMinusAlpha, gradient[i])
                );
            }
        }

        return gradient;
    }

    /// <summary>
    /// Applies softmax function with temperature scaling to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw network outputs before activation.</param>
    /// <param name="temperature">Temperature parameter for softening the distribution.</param>
    /// <returns>Probability distribution summing to 1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax converts raw scores (logits) into probabilities that sum to 1.
    /// Temperature scaling modifies this process:
    /// - First, divide logits by temperature
    /// - Then apply standard softmax: exp(x_i) / sum(exp(x_j))</para>
    ///
    /// <para>We use the "max subtraction trick" for numerical stability to avoid overflow in exp().</para>
    /// </remarks>
    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        // Divide logits by temperature
        var scaledLogits = new T[n];
        for (int i = 0; i < n; i++)
        {
            double val = _numOps.ToDouble(logits[i]) / temperature;
            scaledLogits[i] = _numOps.FromDouble(val);
        }

        // Find max for numerical stability (prevents overflow in exp)
        T maxLogit = scaledLogits[0];
        for (int i = 1; i < n; i++)
        {
            if (_numOps.GreaterThan(scaledLogits[i], maxLogit))
                maxLogit = scaledLogits[i];
        }

        // Compute exp(logit - max) and sum
        T sum = _numOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = _numOps.ToDouble(_numOps.Subtract(scaledLogits[i], maxLogit));
            expValues[i] = _numOps.FromDouble(Math.Exp(val));
            sum = _numOps.Add(sum, expValues[i]);
        }

        // Normalize to get probabilities
        for (int i = 0; i < n; i++)
        {
            result[i] = _numOps.Divide(expValues[i], sum);
        }

        return result;
    }

    /// <summary>
    /// Computes Kullback-Leibler divergence: KL(p || q) = sum(p * log(p / q)).
    /// </summary>
    /// <param name="p">The "true" distribution (teacher soft predictions).</param>
    /// <param name="q">The "approximate" distribution (student soft predictions).</param>
    /// <returns>KL divergence value (always non-negative, 0 when distributions match).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> KL divergence measures how different two probability distributions are.
    /// - KL = 0 means the distributions are identical
    /// - Higher KL means more difference between distributions</para>
    ///
    /// <para>Unlike symmetric distance metrics, KL divergence is asymmetric: KL(p||q) ≠ KL(q||p).
    /// In distillation, we use KL(teacher || student) to make the student match the teacher.</para>
    /// </remarks>
    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = _numOps.Zero;
        const double epsilon = 1e-10; // Small value to avoid log(0)

        for (int i = 0; i < p.Length; i++)
        {
            double pVal = _numOps.ToDouble(p[i]);
            double qVal = _numOps.ToDouble(q[i]);

            if (pVal > epsilon) // Only compute where p is non-zero
            {
                double contrib = pVal * Math.Log(pVal / (qVal + epsilon));
                divergence = _numOps.Add(divergence, _numOps.FromDouble(contrib));
            }
        }

        return divergence;
    }

    /// <summary>
    /// Computes cross-entropy loss: H(true_labels, predictions) = -sum(true_labels * log(predictions)).
    /// </summary>
    /// <param name="predictions">Predicted probability distribution.</param>
    /// <param name="trueLabels">True labels as one-hot vectors.</param>
    /// <returns>Cross-entropy loss value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cross-entropy measures how well predictions match true labels.
    /// - Lower cross-entropy means better predictions
    /// - For perfect predictions, cross-entropy approaches 0
    /// - For completely wrong predictions, cross-entropy becomes large</para>
    ///
    /// <para>Example: If true label is class 1 (one-hot: [0, 1, 0]) and prediction is [0.1, 0.8, 0.1],
    /// cross-entropy = -log(0.8) ≈ 0.22. If prediction were [0.1, 0.3, 0.6], cross-entropy = -log(0.3) ≈ 1.2 (worse).</para>
    /// </remarks>
    private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
    {
        T entropy = _numOps.Zero;
        const double epsilon = 1e-10; // Small value to avoid log(0)

        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = _numOps.ToDouble(predictions[i]);
            double label = _numOps.ToDouble(trueLabels[i]);

            if (label > epsilon) // Only compute where label is non-zero
            {
                double contrib = -label * Math.Log(pred + epsilon);
                entropy = _numOps.Add(entropy, _numOps.FromDouble(contrib));
            }
        }

        return entropy;
    }
}

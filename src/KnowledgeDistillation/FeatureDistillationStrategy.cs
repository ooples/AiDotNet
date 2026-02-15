
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Implements feature-based knowledge distillation (FitNets) where the student learns to match
/// the teacher's intermediate layer representations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> While standard distillation transfers knowledge through final outputs,
/// feature distillation goes deeper by matching intermediate layer activations. This helps the student
/// learn not just what the teacher predicts, but how it thinks.</para>
///
/// <para><b>Why Feature Distillation?</b>
/// - **Better for Different Architectures**: When student and teacher have very different structures
/// - **Richer Knowledge Transfer**: Captures hierarchical feature learning
/// - **Improved Generalization**: Student learns more robust representations
/// - **Complementary to Response Distillation**: Can be combined with standard distillation</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine learning to paint from a master artist. Standard distillation is like copying only the
/// final painting. Feature distillation is like watching the master's brush strokes, color mixing,
/// and layering techniques - learning the process, not just the result.</para>
///
/// <para><b>How It Works:</b>
/// 1. Extract features from a teacher layer (e.g., conv3 in ResNet)
/// 2. Extract features from corresponding student layer
/// 3. Minimize MSE (Mean Squared Error) between them
/// 4. Optionally use a projection layer if dimensions don't match</para>
///
/// <para><b>Common Applications:</b>
/// - ResNet → MobileNet: Match convolutional feature maps
/// - BERT → DistilBERT: Match transformer layer outputs
/// - Teacher and student with different widths/depths</para>
///
/// <para><b>References:</b>
/// - Romero, A., et al. (2014). FitNets: Hints for Thin Deep Nets. arXiv:1412.6550</para>
/// </remarks>
public class FeatureDistillationStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly string[] _layerPairs;
    private readonly double _featureWeight;

    /// <summary>
    /// Initializes a new instance of the FeatureDistillationStrategy class.
    /// </summary>
    /// <param name="layerPairs">Names of layer pairs to match (teacher_layer:student_layer format).
    /// Example: ["conv3:conv2", "conv4:conv3"]</param>
    /// <param name="featureWeight">Weight for feature loss vs. output loss (default 0.5).
    /// Higher values emphasize matching intermediate features.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Layer pairs specify which teacher layers should be matched
    /// to which student layers. Format: "teacher_layer_name:student_layer_name"</para>
    ///
    /// <para>Example usage:
    /// <code>
    /// // Match teacher's layer 3 to student's layer 2, and teacher's 4 to student's 3
    /// var strategy = new FeatureDistillationStrategy&lt;double&gt;(
    ///     layerPairs: new[] { "layer3:layer2", "layer4:layer3" },
    ///     featureWeight: 0.5  // Equal weight to features and outputs
    /// );
    /// </code>
    /// </para>
    ///
    /// <para><b>Tips for choosing layer pairs:</b>
    /// - Match semantically similar layers (similar depth in network)
    /// - Start with 1-2 pairs, add more if needed
    /// - Earlier layers: low-level features (edges, textures)
    /// - Later layers: high-level features (objects, concepts)</para>
    /// </remarks>
    public FeatureDistillationStrategy(string[] layerPairs, double featureWeight = 0.5)
    {
        if (featureWeight < 0 || featureWeight > 1)
            throw new ArgumentException("Feature weight must be between 0 and 1", nameof(featureWeight));

        _numOps = MathHelper.GetNumericOperations<T>();
        Guard.NotNull(layerPairs);
        _layerPairs = layerPairs;
        _featureWeight = featureWeight;

        if (_layerPairs.Length == 0)
            throw new ArgumentException("At least one layer pair must be specified", nameof(layerPairs));

        // Validate layer pair format: must be exactly "teacher_layer:student_layer" with both parts non-empty
        var invalidPairs = _layerPairs.Where(pair =>
        {
            if (string.IsNullOrWhiteSpace(pair))
                return true;

            var parts = pair.Split(':');

            // Must have exactly 2 parts
            if (parts.Length != 2)
                return true;

            // Both parts must be non-empty after trimming
            if (string.IsNullOrWhiteSpace(parts[0]) || string.IsNullOrWhiteSpace(parts[1]))
                return true;

            return false;
        }).ToArray();

        if (invalidPairs.Length > 0)
        {
            var invalidList = string.Join(", ", invalidPairs.Select(p => $"'{p}'"));
            throw new ArgumentException(
                $"Invalid layer pair format: {invalidList}. Expected 'teacher_layer:student_layer' " +
                $"with exactly one colon and both layer names non-empty",
                nameof(layerPairs));
        }
    }

    /// <summary>
    /// Computes the feature matching loss between student and teacher intermediate representations.
    /// </summary>
    /// <param name="teacherFeatureExtractor">Function to extract teacher features for a layer name.</param>
    /// <param name="studentFeatureExtractor">Function to extract student features for a layer name.</param>
    /// <param name="input">Input data for forward pass.</param>
    /// <returns>Mean squared error between matched feature pairs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes how different the student's internal features
    /// are from the teacher's. Lower loss means the student is learning to think like the teacher.</para>
    ///
    /// <para>The loss is computed as:
    /// L_feature = (1/N) × Σ MSE(teacher_features_i, student_features_i)
    /// where N is the number of layer pairs.</para>
    ///
    /// <para>If feature dimensions don't match, consider adding a projection layer
    /// (simple linear transformation) to the student.</para>
    /// </remarks>
    public T ComputeFeatureLoss(
        Func<string, Vector<T>> teacherFeatureExtractor,
        Func<string, Vector<T>> studentFeatureExtractor,
        Vector<T> input)
    {
        if (teacherFeatureExtractor == null) throw new ArgumentNullException(nameof(teacherFeatureExtractor));
        if (studentFeatureExtractor == null) throw new ArgumentNullException(nameof(studentFeatureExtractor));
        if (input == null) throw new ArgumentNullException(nameof(input));

        T totalLoss = _numOps.Zero;

        // Parse layer pairs into structured format using explicit Select
        var parsedPairs = _layerPairs.Select(pair =>
        {
            var parts = pair.Split(':');
            return (TeacherLayer: parts[0].Trim(), StudentLayer: parts[1].Trim());
        });

        foreach (var (teacherLayer, studentLayer) in parsedPairs)
        {
            // Extract features from both models
            var teacherFeatures = teacherFeatureExtractor(teacherLayer);
            var studentFeatures = studentFeatureExtractor(studentLayer);

            // Compute MSE between feature vectors
            var loss = ComputeMSE(studentFeatures, teacherFeatures);
            totalLoss = _numOps.Add(totalLoss, loss);
        }

        // Average across all layer pairs
        totalLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(_layerPairs.Length));

        // Apply feature weight
        totalLoss = _numOps.Multiply(totalLoss, _numOps.FromDouble(_featureWeight));

        return totalLoss;
    }

    /// <summary>
    /// Computes Mean Squared Error between two feature vectors.
    /// </summary>
    /// <param name="studentFeatures">Student's feature vector.</param>
    /// <param name="teacherFeatures">Teacher's feature vector.</param>
    /// <returns>MSE value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> MSE measures the average squared difference between values.
    /// Formula: MSE = (1/n) × Σ (student_i - teacher_i)²</para>
    ///
    /// <para>Why square the differences?
    /// - Large errors are penalized more than small errors
    /// - All differences are positive (no cancellation)
    /// - Mathematically convenient for gradient computation</para>
    /// </remarks>
    private T ComputeMSE(Vector<T> studentFeatures, Vector<T> teacherFeatures)
    {
        if (studentFeatures.Length != teacherFeatures.Length)
        {
            throw new ArgumentException(
                $"Feature dimensions must match. Teacher: {teacherFeatures.Length}, Student: {studentFeatures.Length}. " +
                $"Consider adding a projection layer to the student.");
        }

        T sumSquaredDiff = _numOps.Zero;

        for (int i = 0; i < studentFeatures.Length; i++)
        {
            var diff = _numOps.Subtract(studentFeatures[i], teacherFeatures[i]);
            var squared = _numOps.Multiply(diff, diff);
            sumSquaredDiff = _numOps.Add(sumSquaredDiff, squared);
        }

        return _numOps.Divide(sumSquaredDiff, _numOps.FromDouble(studentFeatures.Length));
    }

    /// <summary>
    /// Computes the gradient of feature loss for backpropagation.
    /// </summary>
    /// <param name="studentFeatures">Student's feature vector.</param>
    /// <param name="teacherFeatures">Teacher's feature vector.</param>
    /// <returns>Gradient vector for backpropagation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The gradient of MSE is simple:
    /// ∂MSE/∂student = (2/n) × (student - teacher)</para>
    ///
    /// <para>This tells us: if student feature is too high, decrease it; if too low, increase it.</para>
    /// </remarks>
    public Vector<T> ComputeFeatureGradient(Vector<T> studentFeatures, Vector<T> teacherFeatures)
    {
        if (studentFeatures.Length != teacherFeatures.Length)
            throw new ArgumentException("Feature dimensions must match");

        int n = studentFeatures.Length;
        var gradient = new Vector<T>(n);
        var scale = _numOps.FromDouble((2.0 / n) * _featureWeight);

        for (int i = 0; i < n; i++)
        {
            var diff = _numOps.Subtract(studentFeatures[i], teacherFeatures[i]);
            gradient[i] = _numOps.Multiply(scale, diff);
        }

        return gradient;
    }
}

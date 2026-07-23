using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Support Vector Machine classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Support Vector Machines (SVMs) find the optimal hyperplane that maximizes the margin
/// between classes. They are particularly effective in high-dimensional spaces and
/// can handle non-linear classification using kernel functions.
/// </para>
/// <para><b>For Beginners:</b> SVMs are like drawing the best possible line between classes!
///
/// Imagine you have red and blue dots on paper. You want to draw a line that:
/// 1. Separates the red dots from the blue dots
/// 2. Stays as far as possible from both sets of dots
///
/// The "margin" is the gap between the line and the nearest dots on each side.
/// SVMs find the line that maximizes this margin.
///
/// Key concepts:
/// - C parameter: How much to penalize misclassifications (higher = stricter)
/// - Kernel: How to measure similarity between points (linear, RBF, polynomial)
/// - Support vectors: The dots closest to the line that define its position
///
/// SVMs work great when:
/// - You have clear separation between classes
/// - You have many features but not tons of samples
/// - You need a robust classifier
/// </para>
/// </remarks>
public class SVMOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization parameter C.
    /// </summary>
    /// <value>
    /// The regularization strength. Default is 1.0.
    /// </value>
    /// <remarks>
    /// <para>
    /// C controls the trade-off between achieving a low training error and a low
    /// testing error (generalization). Higher C means less regularization and
    /// the model will try harder to classify all training points correctly.
    /// </para>
    /// <para><b>For Beginners:</b> C controls how strictly the SVM fits the training data.
    ///
    /// - C = 0.01: Very flexible boundary, allows many misclassifications
    /// - C = 1.0: Balanced (default), good starting point
    /// - C = 100: Very strict, tries to classify all training points correctly
    ///
    /// If your model overfits (high training accuracy, low test accuracy), try lower C.
    /// If your model underfits (low training accuracy), try higher C.
    /// </para>
    /// </remarks>
    public double C { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the kernel type.
    /// </summary>
    /// <value>
    /// The kernel function to use. Default is RBF.
    /// </value>
    /// <remarks>
    /// <para>
    /// The kernel determines how similarity between samples is computed.
    /// Different kernels can capture different types of relationships.
    /// </para>
    /// <para><b>For Beginners:</b> Kernels let SVMs work with non-linear boundaries!
    ///
    /// - Linear: Straight line/plane separation (fast, works when data is linearly separable)
    /// - RBF (Radial Basis Function): Flexible curves (most popular, handles complex patterns)
    /// - Polynomial: Curved boundaries with polynomial degree
    /// - Sigmoid: S-shaped curves (similar to neural networks)
    ///
    /// Start with RBF - it works well for most problems.
    /// Use Linear if you have many features or need speed.
    /// </para>
    /// </remarks>
    public KernelType Kernel { get; set; } = KernelType.RBF;

    /// <summary>
    /// Gets or sets the kernel coefficient gamma.
    /// </summary>
    /// <value>
    /// The gamma value, or null for "scale" (1 / (n_features * variance)). Default is null.
    /// </value>
    /// <remarks>
    /// <para>
    /// Gamma defines how far the influence of a single training example reaches.
    /// Low gamma means 'far' (smoother decision boundary), high gamma means 'close'
    /// (more complex, wiggly boundary).
    /// </para>
    /// <para><b>For Beginners:</b> Gamma controls the "reach" of each training point.
    ///
    /// - Low gamma (0.001): Each point influences a large area (smooth boundary)
    /// - High gamma (10.0): Each point only influences nearby points (complex boundary)
    ///
    /// If null, gamma is set automatically based on your data.
    /// Too high gamma = overfitting
    /// Too low gamma = underfitting
    /// </para>
    /// </remarks>
    public double? Gamma { get; set; } = null;

    /// <summary>
    /// Gets or sets the degree for polynomial kernel.
    /// </summary>
    /// <value>
    /// The polynomial degree. Default is 3.
    /// </value>
    /// <remarks>
    /// Only used when Kernel is Polynomial.
    /// Higher degrees can model more complex relationships but may overfit.
    /// </remarks>
    public int Degree { get; set; } = 3;

    /// <summary>
    /// Gets or sets the independent term (coef0) in kernel function.
    /// </summary>
    /// <value>
    /// The coefficient. Default is 0.0.
    /// </value>
    /// <remarks>
    /// Used in polynomial and sigmoid kernels.
    /// </remarks>
    public double Coef0 { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the tolerance for stopping criterion.
    /// </summary>
    /// <value>
    /// The tolerance. Default is 0.001.
    /// </value>
    /// <remarks>
    /// <para>
    /// The solver iterates until the change in the objective function is below
    /// this threshold. Lower values mean more precise solutions but longer training.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    /// <value>
    /// The maximum iterations, or -1 for unlimited. Default is 1000.
    /// </value>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use shrinking heuristic.
    /// </summary>
    /// <value>
    /// True to use shrinking (default), false otherwise.
    /// </value>
    /// <remarks>
    /// Shrinking can speed up training by focusing on active support vectors.
    /// It usually helps but can sometimes slow things down.
    /// </remarks>
    public bool Shrinking { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to calculate probability estimates.
    /// </summary>
    /// <value>
    /// True to enable probability estimation. Default is false.
    /// </value>
    /// <remarks>
    /// <para>
    /// When true, the SVM will be trained to output probability estimates.
    /// This uses Platt scaling and requires extra cross-validation during training.
    /// </para>
    /// <para><b>For Beginners:</b> Should the SVM output probabilities?
    ///
    /// With Probability = true:
    /// - PredictProbabilities() returns actual probabilities (0.0 to 1.0)
    /// - Training takes longer (needs calibration)
    ///
    /// With Probability = false:
    /// - PredictProbabilities() returns estimates based on decision function
    /// - Faster training
    ///
    /// Enable this if you need calibrated probability outputs.
    /// </para>
    /// </remarks>
    public bool Probability { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use one-vs-rest or one-vs-one for multi-class.
    /// </summary>
    /// <value>
    /// True for one-vs-rest, false for one-vs-one. Default is false (OvO).
    /// </value>
    /// <remarks>
    /// <para>
    /// For multi-class classification:
    /// - One-vs-One (OvO): Train n*(n-1)/2 classifiers, one for each pair of classes
    /// - One-vs-Rest (OvR): Train n classifiers, each separating one class from all others
    /// </para>
    /// <para><b>For Beginners:</b> How to handle more than 2 classes?
    ///
    /// OvO (default):
    /// - Creates classifiers for every pair of classes
    /// - More classifiers but each is simpler
    /// - Often works better in practice
    ///
    /// OvR:
    /// - Creates one classifier per class
    /// - Each classifier separates one class from "everything else"
    /// - Fewer classifiers but each may be harder to train
    /// </para>
    /// </remarks>
    public bool OneVsRest { get; set; } = false;

    /// <summary>
    /// Gets or sets the cache size in MB for kernel computations.
    /// </summary>
    /// <value>
    /// The cache size in megabytes. Default is 200.
    /// </value>
    public double CacheSize { get; set; } = 200.0;
}

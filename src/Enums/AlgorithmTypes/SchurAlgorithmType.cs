namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for computing the Schur decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Schur decomposition is an important way to break down a square matrix into simpler parts 
/// that are easier to work with. It's like taking a complex machine and disassembling it into basic components.
/// 
/// Specifically, the Schur decomposition of a matrix A gives you:
/// A = QTQ*
/// 
/// Where:
/// - Q is a unitary matrix (a special kind of matrix where Q* × Q = I, the identity matrix)
/// - T is an upper triangular matrix (has zeros below the diagonal)
/// - Q* is the conjugate transpose of Q (flip the matrix over its diagonal and take complex conjugates)
/// 
/// In simpler terms:
/// 1. Q represents a change in coordinate system (like rotating a graph's axes)
/// 2. T represents a simplified version of the original transformation
/// 3. Q* represents changing back to the original coordinate system
/// 
/// Why is the Schur decomposition important in AI and machine learning?
/// 
/// 1. Eigenvalue Calculations: It helps find eigenvalues efficiently, which are crucial for techniques like 
///    Principal Component Analysis (PCA)
/// 
/// 2. Matrix Functions: Makes it easier to compute functions of matrices (like matrix exponentials) used in 
///    certain neural network architectures
/// 
/// 3. Stability Analysis: Helps analyze the stability of dynamical systems and recurrent neural networks
/// 
/// 4. Dimensionality Reduction: Contributes to techniques that reduce the complexity of high-dimensional data
/// 
/// 5. Solving Systems: Can be used to efficiently solve certain types of linear systems
/// 
/// This enum specifies which specific algorithm to use for computing the Schur decomposition, as different 
/// methods have different performance characteristics depending on the matrix properties.
/// </para>
/// </remarks>
public enum SchurAlgorithmType
{
    /// <summary>
    /// Uses the Francis QR algorithm with implicit shifts to compute the Schur decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Francis algorithm is a sophisticated method that efficiently computes the Schur 
    /// decomposition by using clever mathematical shortcuts.
    /// 
    /// Imagine you're trying to solve a maze: instead of checking every possible path (which would take forever), 
    /// you use a strategy that lets you eliminate many paths at once. The Francis algorithm does something similar 
    /// with matrices.
    /// 
    /// The key features of the Francis algorithm:
    /// 
    /// 1. It uses "shifts" to accelerate convergence - this means it makes educated guesses about the eigenvalues 
    ///    and uses these guesses to speed up the process
    /// 
    /// 2. It works with "bulges" that move through the matrix, gradually transforming it into the desired form
    /// 
    /// 3. It's much faster than basic QR iteration, especially for large matrices
    /// 
    /// 4. It's the standard algorithm used in professional numerical libraries
    /// 
    /// 5. It handles both real and complex matrices efficiently
    /// 
    /// In machine learning applications, this efficient algorithm enables faster training of models that rely on 
    /// eigenvalue decompositions, speeds up covariance matrix analysis in high-dimensional data, and makes certain 
    /// types of neural network operations more practical for large-scale problems.
    /// </para>
    /// </remarks>
    Francis,

    /// <summary>
    /// Uses an implicit double-shift QR algorithm to compute the Schur decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Implicit algorithm is a variation that focuses on numerical stability and efficiency 
    /// by avoiding explicit calculations of certain intermediate results.
    /// 
    /// Think of it like mental math: instead of writing down every step when calculating 5×18, you might think 
    /// "5×20=100, then subtract 5×2=10, so the answer is 90." You're implicitly handling the calculation without 
    /// explicitly writing out each step.
    /// 
    /// The Implicit algorithm:
    /// 
    /// 1. Reduces roundoff errors by minimizing the number of explicit calculations
    /// 
    /// 2. Uses mathematical properties to perform multiple operations at once
    /// 
    /// 3. Is particularly good for matrices with clustered eigenvalues (values that are close together)
    /// 
    /// 4. Maintains better numerical precision for ill-conditioned problems
    /// 
    /// 5. Often uses double shifts (handling pairs of eigenvalues at once) for real matrices
    /// 
    /// In machine learning contexts, this algorithm is valuable when working with sensitive data where small 
    /// numerical errors could lead to significantly different results, or when analyzing systems where eigenvalues 
    /// are very close together, which happens frequently in certain types of network analysis and signal processing 
    /// applications.
    /// </para>
    /// </remarks>
    Implicit,

    /// <summary>
    /// Uses the basic QR iteration algorithm to compute the Schur decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The QR algorithm is the fundamental approach to computing the Schur decomposition through 
    /// repeated QR decompositions and recombinations.
    /// 
    /// Imagine you're kneading dough: you fold it, roll it out, fold it again, and so on. Each time, the dough 
    /// gets closer to the consistency you want. The QR algorithm repeatedly transforms the matrix in a similar way, 
    /// getting closer to the triangular form with each iteration.
    /// 
    /// The basic process works like this:
    /// 1. Start with your matrix A0
    /// 2. Compute the QR decomposition: A0 = Q1R1
    /// 3. Form a new matrix by multiplying in the reverse order: A1 = R1Q1
    /// 4. Repeat steps 2-3 until the matrix converges to triangular form
    /// 
    /// The QR algorithm:
    /// 
    /// 1. Is conceptually simpler than the Francis algorithm
    /// 
    /// 2. Is easier to implement and understand
    /// 
    /// 3. Works well for small matrices and educational purposes
    /// 
    /// 4. Converges more slowly than shifted variants (like Francis)
    /// 
    /// 5. Provides a good foundation for understanding more advanced methods
    /// 
    /// In machine learning applications, understanding the basic QR algorithm helps build intuition about how 
    /// eigenvalues are computed in practice, which is important when implementing custom algorithms or when 
    /// troubleshooting issues related to matrix decompositions in data analysis pipelines.
    /// </para>
    /// </remarks>
    QR
}

namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for computing eigenvalues and eigenvectors of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Eigenvalues and eigenvectors are special numbers and vectors associated with a matrix 
/// that help us understand the matrix's fundamental properties and behavior.
/// 
/// Think of a matrix as a transformation that changes the position, scale, or rotation of points in space. 
/// Most vectors will change both their direction and length when this transformation is applied. However, 
/// eigenvectors are special vectors that only change in length (but keep their direction) when the 
/// transformation is applied. The eigenvalue tells us how much the eigenvector is stretched or compressed.
/// 
/// Why are these important in AI and machine learning?
/// 
/// 1. Principal Component Analysis (PCA): A popular technique for dimensionality reduction that uses 
///    eigenvectors to find the most important features in your data.
/// 
/// 2. Recommendation Systems: Eigenvalue methods help identify patterns in user preferences.
/// 
/// 3. Image Processing: Facial recognition and image compression often use eigenvalue techniques.
/// 
/// 4. Natural Language Processing: Some algorithms use eigenvalues to analyze relationships between words.
/// 
/// This enum lists different mathematical approaches to find these eigenvalues and eigenvectors, each with 
/// its own advantages depending on the specific problem you're solving.
/// </para>
/// </remarks>
public enum EigenAlgorithmType
{
    /// <summary>
    /// Uses QR decomposition to find eigenvalues and eigenvectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The QR algorithm is one of the most widely used methods for finding all eigenvalues 
    /// and eigenvectors of a matrix.
    /// 
    /// It works by repeatedly decomposing the matrix into a product of two matrices (Q and R) and then 
    /// recombining them in the opposite order. After many iterations, the matrix gradually transforms into 
    /// a form where the eigenvalues become visible.
    /// 
    /// Imagine kneading dough repeatedly - folding it over and pressing it down again and again. Eventually, 
    /// the dough reaches the right consistency. Similarly, the QR algorithm repeatedly "kneads" the matrix 
    /// until its eigenvalues become apparent.
    /// 
    /// This method is very reliable and can find all eigenvalues with good accuracy. It's the default choice 
    /// in many software packages when you need all eigenvalues and eigenvectors of a matrix.
    /// </para>
    /// </remarks>
    QR,

    /// <summary>
    /// Uses the power iteration method to find the dominant eigenvalue and eigenvector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Power Iteration method is a simple approach that finds the largest eigenvalue 
    /// (in absolute value) and its corresponding eigenvector.
    /// 
    /// It works by repeatedly multiplying the matrix by a vector. After many iterations, this vector will 
    /// point in the direction of the dominant eigenvector.
    /// 
    /// Think of it like water flowing downhill - no matter where the water starts, it will eventually find 
    /// the steepest path down. Similarly, power iteration will eventually find the "strongest direction" 
    /// of the matrix transformation.
    /// 
    /// This method is very simple to implement and understand. It's particularly useful when you only need 
    /// the largest eigenvalue (like in some network analysis or ranking algorithms) rather than all eigenvalues. 
    /// However, it can be slow to converge for certain types of matrices.
    /// </para>
    /// </remarks>
    PowerIteration,

    /// <summary>
    /// Uses the Jacobi eigenvalue algorithm to find all eigenvalues and eigenvectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi method finds eigenvalues by gradually making the off-diagonal elements 
    /// of the matrix smaller and smaller through a series of rotations.
    /// 
    /// Imagine you have a wobbly table and you're trying to make it stable by adjusting one leg at a time. 
    /// Each adjustment (rotation) reduces some of the wobble. After many adjustments, the table becomes 
    /// stable. In the Jacobi method, each rotation makes the matrix closer to a diagonal form, where the 
    /// eigenvalues appear on the diagonal.
    /// 
    /// This method is particularly good for symmetric matrices (matrices that are mirror images across their 
    /// diagonal) and is known for its accuracy. It's often used in structural engineering, vibration analysis, 
    /// and quantum mechanics calculations where high precision is required.
    /// 
    /// While it might not be as fast as some other methods for very large matrices, its reliability and 
    /// accuracy make it a popular choice for many applications.
    /// </para>
    /// </remarks>
    Jacobi
}

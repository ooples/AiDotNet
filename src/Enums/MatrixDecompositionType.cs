namespace AiDotNet.Enums;

/// <summary>
/// Specifies different methods for breaking down (decomposing) matrices into simpler components.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Matrix decomposition is like breaking a complex puzzle into simpler pieces 
/// that are easier to work with. In mathematics, we often need to break down complex matrices 
/// (grids of numbers) into simpler components to solve problems more efficiently.
/// 
/// Think of it as:
/// - Breaking down a complex number like 15 into its factors 3 × 5
/// - Disassembling a complicated machine into its basic parts
/// - Converting a difficult problem into several easier ones
/// 
/// Matrix decompositions are important in AI and machine learning for:
/// - Solving systems of equations efficiently
/// - Reducing the dimensionality of data
/// - Finding patterns in data
/// - Making certain calculations faster and more stable
/// - Enabling specific types of analysis
/// 
/// Different decomposition methods have different strengths, weaknesses, and use cases.
/// </para>
/// </remarks>
public enum MatrixDecompositionType
{
    /// <summary>
    /// A method for solving systems of linear equations using determinants.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cramer's rule uses determinants (special numbers calculated from matrices) 
    /// to find the solution to a system of equations.
    /// 
    /// Think of it as:
    /// - A formula-based approach rather than a step-by-step process
    /// - Finding the value of each variable independently
    /// 
    /// Best used for:
    /// - Small systems of equations
    /// - Theoretical proofs
    /// - Understanding the relationship between solutions and matrix properties
    /// 
    /// Not typically used for large systems due to computational inefficiency.
    /// </para>
    /// </remarks>
    Cramer,

    /// <summary>
    /// A decomposition for symmetric, positive-definite matrices into a lower triangular matrix and its transpose.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cholesky decomposition breaks a special type of matrix (symmetric and positive-definite) 
    /// into a simpler form: a lower triangular matrix multiplied by its mirror image.
    /// 
    /// Think of it as:
    /// - Finding the "square root" of a matrix
    /// - Breaking a complex structure into matching halves
    /// 
    /// Best used for:
    /// - Monte Carlo simulations
    /// - Efficient solution of linear systems
    /// - Numerical optimization
    /// - When working with covariance matrices in statistics
    /// 
    /// It's about twice as efficient as LU decomposition for applicable matrices.
    /// </para>
    /// </remarks>
    Cholesky,

    /// <summary>
    /// A method for converting a set of vectors into an orthogonal or orthonormal set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gram-Schmidt takes a set of vectors that might be pointing in various directions 
    /// and converts them into a new set of vectors that are all perpendicular (orthogonal) to each other.
    /// 
    /// Think of it as:
    /// - Reorganizing a messy set of directions into clean north/south/east/west directions
    /// - Creating a better coordinate system from an awkward one
    /// 
    /// Best used for:
    /// - Creating orthogonal basis vectors
    /// - QR decomposition
    /// - Solving least squares problems
    /// - Feature engineering in machine learning
    /// </para>
    /// </remarks>
    GramSchmidt,

    /// <summary>
    /// Decomposes a matrix into a product of lower and upper triangular matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LU decomposition breaks a matrix into a product of two triangular matrices - 
    /// one with non-zero entries below the diagonal (L) and one with non-zero entries above the diagonal (U).
    /// 
    /// Think of it as:
    /// - Breaking a complex transformation into two simpler sequential steps
    /// - Factoring a number into its prime components, but for matrices
    /// 
    /// Best used for:
    /// - Solving linear systems efficiently
    /// - Computing determinants
    /// - Matrix inversion
    /// - When you need to solve multiple systems with the same coefficient matrix
    /// </para>
    /// </remarks>
    Lu,

    /// <summary>
    /// Decomposes a matrix into an orthogonal matrix Q and an upper triangular matrix R.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> QR decomposition breaks a matrix into a product of an orthogonal matrix Q 
    /// (which preserves lengths and angles) and an upper triangular matrix R.
    /// 
    /// Think of it as:
    /// - First rotating your data (Q) and then stretching/shearing it (R)
    /// - Creating a clean coordinate system and then describing transformations in that system
    /// 
    /// Best used for:
    /// - Solving least squares problems
    /// - Eigenvalue algorithms
    /// - Linear regression
    /// - Computing orthonormal bases
    /// </para>
    /// </remarks>
    Qr,

    /// <summary>
    /// Singular Value Decomposition - factorizes a matrix into three components representing rotation, scaling, and another rotation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SVD is one of the most powerful matrix decompositions that breaks any matrix into 
    /// three components: U (rotation/reflection), S (scaling), and V* (another rotation/reflection).
    /// 
    /// Think of it as:
    /// - Revealing the underlying structure and important directions in your data
    /// - Finding the "natural axes" of your data and how much it stretches along each axis
    /// - Like a Swiss Army knife for matrix problems - extremely versatile
    /// 
    /// Best used for:
    /// - Dimensionality reduction (like in PCA)
    /// - Image compression
    /// - Recommendation systems
    /// - Noise reduction
    /// - Finding the "rank" or effective complexity of a dataset
    /// - Solving ill-conditioned linear systems
    /// 
    /// SVD is computationally intensive but extremely powerful and stable.
    /// </para>
    /// </remarks>
    Svd,

    /// <summary>
    /// Transforms a matrix into a form that simplifies certain calculations in least squares problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normal decomposition transforms a matrix into a form that's useful for solving 
    /// least squares problems (finding the best fit for overdetermined systems).
    /// 
    /// Think of it as:
    /// - Rearranging an equation to make it easier to solve
    /// - Converting a complex fitting problem into a simpler form
    /// 
    /// Best used for:
    /// - Linear regression problems
    /// - When you have more equations than unknowns
    /// - Finding approximate solutions when exact solutions don't exist
    /// </para>
    /// </remarks>
    Normal,

    /// <summary>
    /// A variant of QR decomposition that produces a lower triangular matrix L and an orthogonal matrix Q.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LQ decomposition is like QR decomposition but in reverse order - 
    /// it breaks a matrix into a lower triangular matrix L and an orthogonal matrix Q.
    /// 
    /// Think of it as:
    /// - First stretching/shearing your data (L) and then rotating it (Q)
    /// - The "mirror image" of QR decomposition
    /// 
    /// Best used for:
    /// - Certain types of least squares problems
    /// - Some signal processing applications
    /// - When the structure of your problem makes LQ more convenient than QR
    /// </para>
    /// </remarks>
    Lq,

    /// <summary>
    /// A decomposition for complex symmetric matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Takagi decomposition is a specialized decomposition for complex symmetric matrices, 
    /// expressing them in terms of a unitary matrix and a diagonal matrix.
    /// 
    /// Think of it as:
    /// - Finding a special coordinate system where a complex transformation becomes simple
    /// - Breaking down a complex operation into simple scaling operations
    /// 
    /// Best used for:
    /// - Quantum mechanics calculations
    /// - Certain types of optimization problems
    /// - When working with complex symmetric matrices
    /// </para>
    /// </remarks>
    Takagi,

    /// <summary>
    /// Transforms a matrix into Hessenberg form, which is nearly triangular.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hessenberg decomposition transforms a matrix into a form that's almost triangular - 
    /// it has zeros below the first subdiagonal.
    /// 
    /// Think of it as:
    /// - A stepping stone toward finding eigenvalues
    /// - Simplifying a matrix to make further calculations easier
    /// 
    /// Best used for:
    /// - As a preliminary step in eigenvalue algorithms
    /// - Reducing computational complexity in matrix operations
    /// - Making certain iterative methods converge faster
    /// </para>
    /// </remarks>
    Hessenberg,

    /// <summary>
    /// Decomposes a matrix into a product involving an orthogonal matrix and a quasi-triangular matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Schur decomposition expresses a matrix as a product of an orthogonal matrix 
    /// and an upper triangular (or nearly triangular) matrix.
    /// 
    /// Think of it as:
    /// - Finding a coordinate system where a transformation becomes as simple as possible
    /// - Revealing the essential structure of a linear transformation
    /// 
    /// Best used for:
    /// - Computing matrix functions
    /// - Stability analysis
    /// - As part of eigenvalue algorithms
    /// - Understanding the structure of linear transformations
    /// </para>
    /// </remarks>
    Schur,

    /// <summary>
    /// Decomposes a matrix in terms of its eigenvalues and eigenvectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigendecomposition breaks a matrix into its eigenvalues (special scaling factors) 
    /// and eigenvectors (special directions that maintain their orientation when transformed).
    /// 
    /// Think of it as:
    /// - Finding the natural vibration modes of a system
    /// - Identifying the principal directions and stretching factors of a transformation
    /// - Discovering the "natural" coordinate system for a transformation
    /// 
    /// Best used for:
    /// - Principal Component Analysis (PCA)
    /// - Vibration analysis
    /// - Stability analysis
    /// - Quantum mechanics
    /// - Understanding the fundamental behavior of a linear transformation
    /// 
    /// Eigendecomposition only works for diagonalizable matrices.
    /// </para>
    /// </remarks>
    Eigen,

    /// <summary>
    /// Decomposes a matrix into a product of a unitary matrix and a positive semi-definite Hermitian matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Polar decomposition separates a matrix into a rotation/reflection component 
    /// and a stretching/scaling component.
    /// 
    /// Think of it as:
    /// - Breaking down a transformation into "what direction it rotates" and "how much it stretches"
    /// - Similar to writing a complex number in polar form (magnitude and angle)
    /// 
    /// Best used for:
    /// - Computer graphics and animation
    /// - Finding the nearest orthogonal matrix to a given matrix
    /// - Analyzing deformations in physics
    /// - Understanding the geometric meaning of a transformation
    /// </para>
    /// </remarks>
    Polar,

    /// <summary>
    /// Transforms a matrix into a tridiagonal form (non-zero elements only on the main diagonal and the diagonals above and below it).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tridiagonal decomposition converts a matrix into a special form where only three diagonals 
    /// contain non-zero values - the main diagonal and the ones directly above and below it.
    /// 
    /// Think of it as:
    /// - Simplifying a complex matrix into a much simpler form that's easier to work with
    /// - Converting a dense matrix (many non-zero values) into a sparse one (mostly zeros)
    /// - Creating a "band" of values along the diagonal
    /// 
    /// Best used for:
    /// - Solving certain types of differential equations
    /// - Eigenvalue calculations for symmetric matrices
    /// - Efficient storage and computation for large matrices
    /// - Numerical simulations in physics and engineering
    /// 
    /// Tridiagonal systems can be solved very efficiently using specialized algorithms.
    /// </para>
    /// </remarks>
    Tridiagonal,

    /// <summary>
    /// Transforms a matrix into a bidiagonal form (non-zero elements only on the main diagonal and either the diagonal above or below it).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bidiagonal decomposition converts a matrix into an even simpler form than tridiagonal - 
    /// it has non-zero values only on the main diagonal and one adjacent diagonal (either above or below).
    /// 
    /// Think of it as:
    /// - An even more streamlined version of a tridiagonal matrix
    /// - A matrix where information is concentrated in just two diagonals
    /// - A stepping stone toward computing the SVD
    /// 
    /// Best used for:
    /// - As an intermediate step in computing Singular Value Decomposition
    /// - Certain numerical algorithms that benefit from this simplified structure
    /// - Efficient storage and computation for specific types of problems
    /// </para>
    /// </remarks>
    Bidiagonal,

    /// <summary>
    /// Decomposes a symmetric matrix into the product U·D·Uᵀ, where U is upper triangular with 1s on the diagonal and D is diagonal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> UDU decomposition breaks a symmetric matrix into three parts: 
    /// an upper triangular matrix U with 1s on its diagonal, a diagonal matrix D, and the transpose of U.
    /// 
    /// Think of it as:
    /// - A specialized version of LU decomposition for symmetric matrices
    /// - Breaking a complex transformation into simpler components
    /// - A way to efficiently store and work with symmetric matrices
    /// 
    /// Best used for:
    /// - Solving symmetric linear systems
    /// - Stability analysis in numerical methods
    /// - Efficient implementation of certain algorithms for symmetric matrices
    /// - When memory efficiency is important
    /// 
    /// The UDU decomposition is numerically stable and memory-efficient for symmetric matrices.
    /// </para>
    /// </remarks>
    Udu,

    /// <summary>
    /// Decomposes a symmetric matrix into the product L·D·Lᵀ, where L is lower triangular with 1s on the diagonal and D is diagonal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LDL decomposition is similar to UDU but uses lower triangular matrices instead of upper ones.
    /// It breaks a symmetric matrix into a lower triangular matrix L with 1s on its diagonal,
    /// a diagonal matrix D, and the transpose of L.
    ///
    /// Think of it as:
    /// - A variant of Cholesky decomposition that avoids square roots
    /// - A way to break down symmetric matrices into simpler components
    /// - A more numerically stable alternative to certain other decompositions
    ///
    /// Best used for:
    /// - Solving symmetric linear systems efficiently
    /// - When working with indefinite matrices (where Cholesky might fail)
    /// - Implementing certain optimization algorithms
    /// - When computational stability is important
    ///
    /// LDL decomposition is often more efficient than LU for symmetric matrices and more stable than Cholesky for indefinite matrices.
    /// </para>
    /// </remarks>
    Ldl,

    /// <summary>
    /// Non-negative Matrix Factorization - decomposes a non-negative matrix into two non-negative matrices W and H.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NMF breaks down a matrix containing only non-negative values (zero or positive)
    /// into two simpler non-negative matrices. It's particularly useful when negative values don't make sense
    /// in your domain (like pixel intensities, word frequencies, or probabilities).
    ///
    /// Think of it as:
    /// - Finding hidden patterns or topics in your data
    /// - Breaking down complex data into interpretable parts
    /// - Like SVD but maintaining non-negativity, which often leads to more interpretable results
    ///
    /// Best used for:
    /// - Topic modeling and text mining (finding themes in documents)
    /// - Image processing and feature extraction
    /// - Recommendation systems (collaborative filtering)
    /// - Audio source separation
    /// - Bioinformatics and gene expression analysis
    /// - Any domain where negative values are meaningless
    ///
    /// NMF often produces more interpretable results than methods that allow negative values
    /// because the non-negativity constraint leads to parts-based representations.
    /// </para>
    /// </remarks>
    Nmf,

    /// <summary>
    /// Independent Component Analysis - separates mixed signals into statistically independent source components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ICA is designed to separate mixed signals into their original independent sources.
    /// The classic example is the "cocktail party problem" - imagine multiple people talking at once and
    /// multiple microphones recording the mixed conversations. ICA can separate out each person's voice.
    ///
    /// Think of it as:
    /// - Un-mixing signals that were combined together
    /// - Finding the hidden independent sources in mixed observations
    /// - Like trying to separate individual instruments from a musical recording
    ///
    /// Best used for:
    /// - Blind source separation (cocktail party problem)
    /// - Brain signal analysis (EEG, MEG, fMRI)
    /// - Removing artifacts from biomedical signals
    /// - Feature extraction where statistical independence is important
    /// - Image separation and analysis
    /// - Financial data analysis
    /// - Telecommunications
    ///
    /// ICA differs from PCA and other methods because it focuses on statistical independence
    /// rather than just uncorrelated patterns, making it powerful for separating truly independent sources.
    /// </para>
    /// </remarks>
    Ica
}

namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of matrices that can be used in mathematical operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A matrix is a rectangular array of numbers arranged in rows and columns.
/// Different types of matrices have special properties that make them useful for specific
/// calculations or applications. This enum lists the various matrix types supported by the library.
/// </para>
/// </remarks>
public enum MatrixType
{
    /// <summary>
    /// Matrix type has not been determined or specified.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// A matrix with the same number of rows and columns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A square matrix has the same number of rows and columns, like a square.
    /// Example: A 3×3 matrix has 3 rows and 3 columns.
    /// </para>
    /// </remarks>
    Square = 1,

    /// <summary>
    /// A matrix where all non-diagonal elements are zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has numbers only along the diagonal from top-left to bottom-right,
    /// with zeros everywhere else.
    /// 
    /// Example:
    /// [5 0 0]
    /// [0 7 0]
    /// [0 0 2]
    /// </para>
    /// </remarks>
    Diagonal = 2,

    /// <summary>
    /// A diagonal matrix where all diagonal elements are 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An identity matrix is a special diagonal matrix with 1s along the diagonal
    /// and 0s everywhere else. It works like the number 1 in multiplication - multiplying any matrix
    /// by the identity matrix leaves it unchanged.
    /// 
    /// Example:
    /// [1 0 0]
    /// [0 1 0]
    /// [0 0 1]
    /// </para>
    /// </remarks>
    Identity = 3,

    /// <summary>
    /// A square matrix that is equal to its transpose (mirror image across the diagonal).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A symmetric matrix is like a mirror image across its diagonal. The value at
    /// position (row 2, column 5) equals the value at (row 5, column 2).
    /// 
    /// Example:
    /// [3 7 2]
    /// [7 4 9]
    /// [2 9 5]
    /// </para>
    /// </remarks>
    Symmetric = 4,

    /// <summary>
    /// A square matrix whose transpose equals its negative.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In a skew-symmetric matrix, the diagonal elements are always zero, and the elements
    /// on opposite sides of the diagonal are negatives of each other. If position (row 1, column 2) has value 5,
    /// then position (row 2, column 1) has value -5.
    /// 
    /// Example:
    /// [ 0  3 -1]
    /// [-3  0  4]
    /// [ 1 -4  0]
    /// </para>
    /// </remarks>
    SkewSymmetric = 5,

    /// <summary>
    /// A matrix where all elements below the main diagonal are zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper triangular matrix has values on or above the diagonal, with all values
    /// below the diagonal being zero. It forms a triangle shape in the upper part of the matrix.
    /// 
    /// Example:
    /// [4 2 1]
    /// [0 7 3]
    /// [0 0 5]
    /// </para>
    /// </remarks>
    UpperTriangular = 6,

    /// <summary>
    /// A matrix where all elements above the main diagonal are zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A lower triangular matrix has values on or below the diagonal, with all values
    /// above the diagonal being zero. It forms a triangle shape in the lower part of the matrix.
    /// 
    /// Example:
    /// [4 0 0]
    /// [2 7 0]
    /// [1 3 5]
    /// </para>
    /// </remarks>
    LowerTriangular = 7,

    /// <summary>
    /// A matrix with a different number of rows and columns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A rectangular matrix has a different number of rows and columns, like a rectangle.
    /// Example: A 2×3 matrix has 2 rows and 3 columns.
    /// </para>
    /// </remarks>
    Rectangular = 8,

    /// <summary>
    /// A matrix where all elements are zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A zero matrix contains only zeros in all positions.
    /// 
    /// Example:
    /// [0 0 0]
    /// [0 0 0]
    /// </para>
    /// </remarks>
    Zero = 9,

    /// <summary>
    /// A diagonal matrix where all diagonal elements are the same value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A scalar matrix is a diagonal matrix where all the diagonal values are the same number.
    /// 
    /// Example:
    /// [5 0 0]
    /// [0 5 0]
    /// [0 0 5]
    /// </para>
    /// </remarks>
    Scalar = 10,

    /// <summary>
    /// A matrix with non-zero elements only on the main diagonal and the diagonal above it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper bidiagonal matrix has values only on the main diagonal and the diagonal
    /// immediately above it, with zeros everywhere else.
    /// 
    /// Example:
    /// [4 7 0]
    /// [0 2 5]
    /// [0 0 9]
    /// </para>
    /// </remarks>
    UpperBidiagonal = 11,

    /// <summary>
    /// A matrix with non-zero elements only on the main diagonal and the diagonal below it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A lower bidiagonal matrix has values only on the main diagonal and the diagonal
    /// immediately below it, with zeros everywhere else.
    /// 
    /// Example:
    /// [4 0 0]
    /// [7 2 0]
    /// [0 5 9]
    /// </para>
    /// </remarks>
    LowerBidiagonal = 12,

    /// <summary>
    /// A matrix with non-zero elements only on the main diagonal and the diagonals immediately above and below it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tridiagonal matrix has values only on the main diagonal and the diagonals
    /// immediately above and below it, with zeros everywhere else.
    /// 
    /// Example:
    /// [4 7 0]
    /// [2 5 8]
    /// [0 3 9]
    /// </para>
    /// </remarks>
    Tridiagonal = 13,

    /// <summary>
    /// A matrix with non-zero elements only on a band centered on the main diagonal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A band matrix has non-zero values only within a certain "band" around the diagonal,
    /// with zeros everywhere else. It's a generalization of tridiagonal matrices to include more diagonals.
    /// 
    /// Example of a band matrix with bandwidth 2:
    /// [5 8 2 0 0]
    /// [1 6 9 3 0]
    /// [0 2 7 1 4]
    /// [0 0 3 8 2]
    /// [0 0 0 4 9]
    /// </para>
    /// </remarks>
    Band = 14,

    /// <summary>
    /// A complex square matrix that equals its own conjugate transpose.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hermitian matrix is the complex number equivalent of a symmetric matrix.
    /// For real matrices, Hermitian and symmetric are the same thing. For complex matrices,
    /// the elements across the diagonal are complex conjugates of each other (same real part,
    /// opposite imaginary part).
    /// 
    /// If position (row 1, column 2) has value 3+4i, then position (row 2, column 1) has value 3-4i.
    /// </para>
    /// </remarks>
    Hermitian = 15,

    /// <summary>
    /// A complex square matrix whose conjugate transpose equals its negative.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A skew-Hermitian matrix is the complex number equivalent of a skew-symmetric matrix.
    /// The diagonal elements are purely imaginary (or zero), and elements across the diagonal are negative
    /// complex conjugates of each other.
    /// 
    /// If position (row 1, column 2) has value 3+4i, then position (row 2, column 1) has value -3+4i.
    /// </para>
    /// </remarks>
    SkewHermitian = 16,

    /// <summary>
    /// A real square matrix whose transpose equals its inverse.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An orthogonal matrix preserves lengths and angles when it multiplies a vector.
    /// It represents rotations and reflections in space. The columns (and rows) of an orthogonal matrix
    /// form a set of perpendicular unit vectors.
    /// 
    /// A key property: multiplying an orthogonal matrix by its transpose gives the identity matrix.
    /// </para>
    /// </remarks>
    Orthogonal = 17,

    /// <summary>
    /// A complex square matrix whose conjugate transpose equals its inverse.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A unitary matrix is the complex number equivalent of an orthogonal matrix.
    /// It preserves lengths and angles in complex vector spaces. For real matrices, unitary and
    /// orthogonal are the same thing.
    /// 
    /// A key property: multiplying a unitary matrix by its conjugate transpose gives the identity matrix.
    /// </para>
    /// </remarks>
    Unitary = 18,

    /// <summary>
    /// A square matrix that doesn't have an inverse.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A singular matrix cannot be inverted (no inverse exists). This happens when
    /// the determinant is zero, which means the matrix equations don't have unique solutions.
    /// 
    /// Think of it as a transformation that "flattens" space in at least one dimension, losing information
    /// in the process, making it impossible to reverse the transformation completely.
    /// </para>
    /// </remarks>
    Singular = 19,

    /// <summary>
    /// A square matrix that has an inverse.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A non-singular matrix can be inverted (has an inverse). This means any transformation
    /// performed by this matrix can be undone. It's like being able to trace your steps backward after
    /// a journey. Non-singular matrices have non-zero determinants.
    /// </para>
    /// </remarks>
    NonSingular = 20,

    /// <summary>
    /// A symmetric matrix where all eigenvalues are positive.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A positive definite matrix has special properties that make it useful in optimization
    /// and statistics. A key characteristic is that when you multiply this matrix with any non-zero vector,
    /// the result is always positive. These matrices represent "bowl-shaped" surfaces that have a clear minimum point.
    /// </para>
    /// </remarks>
    PositiveDefinite = 21,

    /// <summary>
    /// A symmetric matrix where all eigenvalues are non-negative (zero or positive).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A positive semi-definite matrix is similar to a positive definite matrix, but allows
    /// for some eigenvalues to be zero. When multiplied with any vector, the result is always non-negative.
    /// These matrices are common in statistics and machine learning, especially in covariance matrices.
    /// </para>
    /// </remarks>
    PositiveSemiDefinite = 22,

    /// <summary>
    /// A square matrix that represents a projection onto a subspace.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An orthogonal projection matrix "projects" vectors onto a specific subspace.
    /// Think of it like a shadow cast by an object onto a wall - the projection matrix represents
    /// this shadow-casting process. These matrices are used in data analysis to reduce dimensions
    /// while preserving important information.
    /// </para>
    /// </remarks>
    OrthogonalProjection = 23,

    /// <summary>
    /// A matrix that, when multiplied by itself, gives the same matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An idempotent matrix has the property that multiplying it by itself gives the same matrix.
    /// That is, A² = A. This is like a light switch that's already on - flipping it again doesn't change anything.
    /// Projection matrices are examples of idempotent matrices.
    /// </para>
    /// </remarks>
    Idempotent = 24,

    /// <summary>
    /// A matrix that, when multiplied by itself, gives the identity matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An involutory matrix is its own inverse. When you multiply it by itself, you get
    /// the identity matrix. It's like applying a transformation and then applying it again to get back
    /// to where you started. Reflection matrices are common examples.
    /// </para>
    /// </remarks>
    Involutory = 25,

    /// <summary>
    /// A matrix where all elements are non-negative and each row sums to 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A stochastic matrix (also called a probability matrix) has non-negative entries
    /// with each row summing to 1. These matrices represent transition probabilities in Markov chains,
    /// where each row shows the probability of moving from one state to others.
    /// </para>
    /// </remarks>
    Stochastic = 26,

    /// <summary>
    /// A matrix where all elements are non-negative and both rows and columns sum to 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A doubly stochastic matrix has non-negative entries where both every row and every column
    /// sum to 1. These matrices are used in assignment problems and represent balanced distributions.
    /// </para>
    /// </remarks>
    DoublyStochastic = 27,

    /// <summary>
    /// A matrix that has exactly one 1 in each row and each column, with all other elements being 0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A permutation matrix rearranges the order of elements in a vector when multiplied with it.
    /// It's like shuffling a deck of cards in a specific way. Each row and column has exactly one 1, with the rest being 0s.
    /// </para>
    /// </remarks>
    Permutation = 28,

    /// <summary>
    /// A matrix that represents connections in a graph or network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An adjacency matrix represents connections in a network or graph. If element (i,j) is 1,
    /// it means there's a connection from node i to node j. These matrices are used in social networks,
    /// transportation systems, and computer networks to show how elements are connected.
    /// </para>
    /// </remarks>
    Adjacency = 29,

    /// <summary>
    /// A matrix that shows relationships between two types of objects in a graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An incidence matrix shows how objects of one type (like people) relate to objects
    /// of another type (like events). In graph theory, it shows how edges connect to vertices.
    /// For example, if row i has a 1 in column j, it means vertex i is connected to edge j.
    /// </para>
    /// </remarks>
    Incidence = 30,

    /// <summary>
    /// A matrix that represents a graph's connectivity and is used in spectral graph theory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Laplacian matrix combines information about connections and degrees (number of connections)
    /// in a graph. It's used to find important properties of networks, like how well-connected they are
    /// or how information flows through them. These matrices are used in image processing, clustering,
    /// and network analysis.
    /// </para>
    /// </remarks>
    Laplacian = 31,

    /// <summary>
    /// A matrix where each descending diagonal from left to right has constant values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Toeplitz matrix has the same value along each diagonal. The value depends only on
    /// how far the diagonal is from the main diagonal. These matrices appear in signal processing and
    /// when solving certain differential equations.
    /// 
    /// Example:
    /// [a b c d]
    /// [e a b c]
    /// [f e a b]
    /// [g f e a]
    /// </para>
    /// </remarks>
    Toeplitz = 32,

    /// <summary>
    /// A matrix where each anti-diagonal (running from bottom-left to top-right) has constant values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hankel matrix has constant values along anti-diagonals (diagonals that run from
    /// bottom-left to top-right). These matrices appear in control theory and signal processing applications.
    /// 
    /// Example:
    /// [a b c d]
    /// [b c d e]
    /// [c d e f]
    /// [d e f g]
    /// </para>
    /// </remarks>
    Hankel = 33,

    /// <summary>
    /// A special Toeplitz matrix where each row is a cyclic shift of the row above it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A circulant matrix is created by taking the first row and shifting it to the right
    /// in each subsequent row. It's like a circular pattern where elements wrap around. These matrices
    /// are used in signal processing, especially in dealing with periodic signals.
    /// 
    /// Example:
    /// [a b c d]
    /// [d a b c]
    /// [c d a b]
    /// [b c d a]
    /// </para>
    /// </remarks>
    Circulant = 34,

    /// <summary>
    /// A matrix divided into submatrices (blocks) that are treated as single elements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A block matrix is divided into rectangular sections called blocks or submatrices.
    /// This structure makes it easier to work with large matrices by treating each block as a single element.
    /// Block matrices are useful in parallel computing and when dealing with matrices that have a natural
    /// block structure.
    /// </para>
    /// </remarks>
    Block = 35,

    /// <summary>
    /// A matrix where most elements are zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A sparse matrix contains mostly zeros, with relatively few non-zero elements.
    /// These matrices are stored efficiently by only recording the positions and values of non-zero elements.
    /// Sparse matrices are common in large-scale problems like social networks, web links, or 3D simulations.
    /// </para>
    /// </remarks>
    Sparse = 36,

    /// <summary>
    /// A matrix where most elements are non-zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A dense matrix has most of its elements filled with non-zero values.
    /// Unlike sparse matrices, dense matrices don't have a special storage format since most elements
    /// need to be stored anyway. These matrices typically represent fully connected systems or relationships.
    /// </para>
    /// </remarks>
    Dense = 37,

    /// <summary>
    /// A matrix that has been divided into sections for specific mathematical operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A partitioned matrix is divided into sections to make certain calculations easier.
    /// Unlike block matrices (which are about storage and structure), partitioning is about breaking down
    /// a problem into smaller parts. This approach is used in solving systems of equations and in matrix
    /// decomposition methods.
    /// </para>
    /// </remarks>
    Partitioned = 38,

    /// <summary>
    /// A special matrix used in polynomial calculations and control theory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A companion matrix is a special form that represents a polynomial in matrix form.
    /// It's used to find polynomial roots and in control systems. The matrix has a specific pattern with
    /// 1s along the first subdiagonal and the coefficients of a polynomial in the last column.
    /// 
    /// Example for polynomial x³ + 4x² + 5x + 2:
    /// [0 0 -2]
    /// [1 0 -5]
    /// [0 1 -4]
    /// </para>
    /// </remarks>
    Companion = 39,

    /// <summary>
    /// A matrix where each row consists of consecutive powers of a value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Vandermonde matrix is created by taking a set of values and raising each value to 
    /// different powers. For example, if we have values [a, b, c], the matrix would look like:
    /// [1    1    1  ]
    /// [a    b    c  ]
    /// [a²   b²   c² ]
    /// These matrices are used in polynomial interpolation (finding a curve that passes through specific points)
    /// and in coding theory for error correction.
    /// </para>
    /// </remarks>
    Vandermonde = 40,

    /// <summary>
    /// A special matrix where each element is 1 divided by the sum of its row and column indices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Hilbert matrix has elements defined by the formula 1/(i+j-1), where i is the row number
    /// and j is the column number. For example, a 3×3 Hilbert matrix looks like:
    /// [1    1/2   1/3]
    /// [1/2  1/3   1/4]
    /// [1/3  1/4   1/5]
    /// These matrices are famous for being "ill-conditioned," which means small changes in input can cause large
    /// changes in output, making them challenging for numerical calculations.
    /// </para>
    /// </remarks>
    Hilbert = 41,

    /// <summary>
    /// A matrix where each element is 1 divided by the sum of two values from separate arrays.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Cauchy matrix is formed from two sets of numbers [x1, x2, ...] and [y1, y2, ...].
    /// Each element (i,j) equals 1/(x? + y?). These matrices appear in interpolation problems and numerical analysis.
    /// 
    /// Example: If x = [1, 2, 3] and y = [4, 5, 6], the Cauchy matrix would be:
    /// [1/5  1/6  1/7 ]
    /// [1/6  1/7  1/8 ]
    /// [1/7  1/8  1/9 ]
    /// </para>
    /// </remarks>
    Cauchy = 42
}

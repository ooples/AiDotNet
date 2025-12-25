namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a data transformer that can fit to data and transform it.
/// </summary>
/// <remarks>
/// <para>
/// This is the core interface for all preprocessing transformers in AiDotNet.
/// It follows the sklearn-style Fit/Transform pattern where transformers first
/// learn parameters from training data (Fit), then apply transformations (Transform).
/// </para>
/// <para><b>For Beginners:</b> A transformer is like a recipe that:
/// 1. First "learns" from your training data (e.g., calculates mean and std for scaling)
/// 2. Then applies the same recipe to any new data
///
/// This ensures new data is processed exactly the same way as training data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type after transformation.</typeparam>
public interface IDataTransformer<T, TInput, TOutput>
{
    /// <summary>
    /// Gets whether this transformer has been fitted to data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns true after <see cref="Fit"/> or <see cref="FitTransform"/> has been called.
    /// Transform operations require the transformer to be fitted first.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the transformer has "learned"
    /// from training data yet. You must call Fit() before Transform().
    /// </para>
    /// </remarks>
    bool IsFitted { get; }

    /// <summary>
    /// Gets the column indices this transformer operates on.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If null or empty, the transformer operates on all columns.
    /// Otherwise, it only processes the specified column indices.
    /// </para>
    /// <para><b>For Beginners:</b> Some transformers only apply to certain columns.
    /// For example, OneHotEncoder might only encode columns 2 and 5.
    /// If this is null, the transformer applies to all columns.
    /// </para>
    /// </remarks>
    int[]? ColumnIndices { get; }

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some transformers can reverse their transformation (e.g., StandardScaler can
    /// convert scaled values back to original scale). Others cannot (e.g., OneHotEncoder
    /// may lose information about category ordering).
    /// </para>
    /// <para><b>For Beginners:</b> If this is true, you can "undo" the transformation.
    /// This is useful for converting predictions back to the original scale.
    /// </para>
    /// </remarks>
    bool SupportsInverseTransform { get; }

    /// <summary>
    /// Fits the transformer to the training data, learning any parameters needed for transformation.
    /// </summary>
    /// <param name="data">The training data to fit.</param>
    /// <remarks>
    /// <para>
    /// This method learns transformation parameters from the data. For example:
    /// - StandardScaler learns mean and standard deviation
    /// - MinMaxScaler learns min and max values
    /// - OneHotEncoder learns unique categories
    /// </para>
    /// <para><b>For Beginners:</b> Call this once on your training data.
    /// The transformer will remember what it learned and use it for all future transforms.
    /// </para>
    /// </remarks>
    void Fit(TInput data);

    /// <summary>
    /// Transforms the input data using the fitted parameters.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    /// <exception cref="InvalidOperationException">Thrown if Fit() has not been called.</exception>
    /// <remarks>
    /// <para>
    /// Applies the learned transformation to the data. This method can be called
    /// multiple times with different data after fitting.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to transform your test data or new predictions.
    /// It applies the same transformation that was learned from training data.
    /// </para>
    /// </remarks>
    TOutput Transform(TInput data);

    /// <summary>
    /// Fits the transformer and transforms the data in a single step.
    /// </summary>
    /// <param name="data">The data to fit and transform.</param>
    /// <returns>The transformed data.</returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method that combines Fit and Transform.
    /// Use this for training data where you want to fit and transform in one call.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for your training data.
    /// It's equivalent to calling Fit() then Transform(), but more convenient.
    /// </para>
    /// </remarks>
    TOutput FitTransform(TInput data);

    /// <summary>
    /// Reverses the transformation (if supported).
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    /// <exception cref="NotSupportedException">Thrown if inverse transform is not supported.</exception>
    /// <exception cref="InvalidOperationException">Thrown if Fit() has not been called.</exception>
    /// <remarks>
    /// <para>
    /// Converts transformed data back to its original scale. This is useful for
    /// interpreting predictions that were made on scaled data.
    /// </para>
    /// <para><b>For Beginners:</b> If you scaled your target values before training,
    /// use this to convert predictions back to the original scale (like dollars or temperatures).
    /// </para>
    /// </remarks>
    TInput InverseTransform(TOutput data);

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names (optional).</param>
    /// <returns>The output feature names.</returns>
    /// <remarks>
    /// <para>
    /// Returns meaningful names for the output features. For transformers that
    /// change the number of features (like OneHotEncoder or PolynomialFeatures),
    /// this returns names reflecting the new features.
    /// </para>
    /// <para><b>For Beginners:</b> This helps you understand what each column means
    /// after transformation. For example, after OneHotEncoder, you get names like
    /// "color_red", "color_blue" instead of just column numbers.
    /// </para>
    /// </remarks>
    string[] GetFeatureNamesOut(string[]? inputFeatureNames = null);
}

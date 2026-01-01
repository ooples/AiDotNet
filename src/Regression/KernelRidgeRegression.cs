namespace AiDotNet.Regression;

/// <summary>
/// Implements Kernel Ridge Regression, a powerful nonlinear regression technique that combines
/// ridge regression with the kernel trick to capture complex nonlinear relationships.
/// </summary>
/// <remarks>
/// <para>
/// Kernel Ridge Regression extends linear ridge regression by applying the "kernel trick" to implicitly
/// map the input features to a higher-dimensional space without explicitly computing the transformation.
/// This allows the model to capture complex nonlinear relationships while still maintaining the computational
/// efficiency of ridge regression. The regularization parameter (lambda) helps prevent overfitting by
/// penalizing large coefficients.
/// </para>
/// <para><b>For Beginners:</b> Kernel Ridge Regression is like using a special lens that helps see complex patterns in your data.
/// 
/// Regular linear models can only fit straight lines to data, but many real-world relationships aren't straight.
/// Kernel Ridge Regression solves this by:
/// - Using "kernels" to transform your data into a format where complex relationships become simpler
/// - Finding patterns in this transformed space
/// - Adding "ridge" regularization to prevent the model from becoming too complex or overfitting
/// 
/// Think of it like this: If you tried to separate red and blue dots on a sheet of paper with a single line,
/// sometimes it's impossible. But if you could lift some dots off the page (into 3D space), you might be able
/// to separate them with a flat plane. Kernels do something similar - they transform your data so that complex
/// patterns become easier to find.
/// 
/// This technique is particularly good for:
/// - Medium-sized datasets with complex relationships
/// - Problems where the relationship between inputs and outputs is highly nonlinear
/// - When you need both good prediction accuracy and the ability to adjust how much the model fits to noise
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class KernelRidgeRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// The Gram matrix (kernel matrix) that represents pairwise similarities between all training points.
    /// </summary>
    private Matrix<T> _gramMatrix;

    /// <summary>
    /// The dual coefficients used for making predictions.
    /// </summary>
    private Vector<T> _dualCoefficients;

    /// <summary>
    /// Gets the configuration options specific to Kernel Ridge Regression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the configuration options that control the Kernel Ridge Regression
    /// algorithm, such as the regularization parameter (lambda) and the matrix decomposition type used for
    /// solving the linear system.
    /// </para>
    /// </remarks>
    private new KernelRidgeRegressionOptions Options => (KernelRidgeRegressionOptions)base.Options;

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelRidgeRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Kernel Ridge Regression algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Kernel Ridge Regression model with the specified options and regularization
    /// strategy. The options control parameters such as the regularization strength (lambda) and the matrix
    /// decomposition type. If no regularization is specified, no regularization is applied beyond the ridge penalty.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Kernel Ridge Regression model.
    /// 
    /// When creating a model, you need to specify:
    /// - Options: Controls settings like the regularization strength (lambda), which determines how much the model
    ///   balances between fitting the training data perfectly and keeping the model simple
    /// - Regularization: Optional additional method to prevent overfitting
    /// 
    /// Example:
    /// ```csharp
    /// // Create options for Kernel Ridge Regression
    /// var options = new KernelRidgeRegressionOptions { 
    ///     LambdaKRR = 0.1,
    ///     KernelType = KernelType.RBF
    /// };
    /// 
    /// // Create the model
    /// var krr = new KernelRidgeRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public KernelRidgeRegression(KernelRidgeRegressionOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _gramMatrix = Matrix<T>.Empty();
        _dualCoefficients = Vector<T>.Empty();
    }

    /// <summary>
    /// Optimizes the Kernel Ridge Regression model based on the provided training data.
    /// </summary>
    /// <param name="X">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in X.</param>
    /// <remarks>
    /// <para>
    /// This method builds the Kernel Ridge Regression model by computing the Gram (kernel) matrix, which represents
    /// the similarity between all pairs of training samples, adding a ridge penalty to the diagonal for regularization,
    /// and solving for the dual coefficients. These coefficients, along with the support vectors (training samples),
    /// are used for making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method does the actual "learning" for the model.
    /// 
    /// Here's what happens during this process:
    /// 1. The model calculates how similar each training example is to every other training example
    ///    (this creates what's called the "Gram matrix" or "kernel matrix")
    /// 2. It adds the ridge penalty to the diagonal of this matrix (this is the regularization that helps
    ///    prevent overfitting)
    /// 3. It solves a mathematical equation to find the best "dual coefficients" - these determine
    ///    how much each training example influences predictions
    /// 4. It stores these coefficients and the training data (support vectors) for making predictions later
    /// 
    /// This process allows the model to capture complex patterns in your data while still maintaining
    /// good generalization to new, unseen data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        _gramMatrix = new Matrix<T>(n, n);

        // Compute the Gram matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                T value = KernelFunction(X.GetRow(i), X.GetRow(j));
                _gramMatrix[i, j] = value;
                _gramMatrix[j, i] = value;
            }
        }

        // Add ridge penalty to the diagonal (LambdaKRR is the primary regularization for KRR)
        for (int i = 0; i < n; i++)
        {
            _gramMatrix[i, i] = NumOps.Add(_gramMatrix[i, i], NumOps.FromDouble(Options.LambdaKRR));
        }

        // Add additional regularization strength if specified
        var regularizationStrength = Regularization?.GetOptions().Strength ?? 0.0;
        if (regularizationStrength > 0)
        {
            T regTerm = NumOps.FromDouble(regularizationStrength);
            for (int i = 0; i < n; i++)
            {
                _gramMatrix[i, i] = NumOps.Add(_gramMatrix[i, i], regTerm);
            }
        }

        // Solve (K + λI)a = y
        _dualCoefficients = MatrixSolutionHelper.SolveLinearSystem(_gramMatrix, y, Options.DecompositionType);

        // Store X as support vectors for prediction
        SupportVectors = X;
        Alphas = _dualCoefficients;
    }

    /// <summary>
    /// Predicts the target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The feature vector of the sample to predict.</param>
    /// <returns>The predicted value for the input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the target value for a single input feature vector by computing the kernel function
    /// between the input and all support vectors (training samples), weighting the results by the dual coefficients,
    /// and summing them. This is the dual form of the prediction equation for kernel-based methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a new data point.
    /// 
    /// To make a prediction:
    /// 1. The model compares the new input to every training example (support vector)
    ///    using the kernel function to measure similarity
    /// 2. It multiplies each similarity value by the corresponding dual coefficient
    ///    (which was learned during training)
    /// 3. It adds up all these weighted similarity values to get the final prediction
    /// 
    /// This approach allows the model to make predictions based on how similar the new input
    /// is to the training examples, with more influential examples (those with larger dual
    /// coefficients) having a greater impact on the prediction.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        T result = NumOps.Zero;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            Vector<T> supportVector = SupportVectors.GetRow(i);
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i], KernelFunction(input, supportVector)));
        }
        return result;
    }

    /// <summary>
    /// Gets metadata about the Kernel Ridge Regression model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, the regularization parameter (lambda),
    /// and the regularization type. This information can be useful for model management, comparison, and documentation
    /// purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your model configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Kernel Ridge Regression)
    /// - The lambda value (regularization strength)
    /// - The type of regularization applied
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = krr.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Lambda: {metadata.AdditionalInfo["LambdaKRR"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["LambdaKRR"] = Options.LambdaKRR;
        metadata.AdditionalInfo["RegularizationType"] = Regularization.GetType().Name;

        return metadata;
    }

    /// <summary>
    /// Gets the model type of the Kernel Ridge Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.KernelRidgeRegression;
    }

    /// <summary>
    /// Serializes the Kernel Ridge Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Kernel Ridge Regression model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the base class data, model-specific options like
    /// the regularization parameter (lambda), the Gram matrix, and the dual coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - The model's settings (like the lambda regularization parameter)
    /// - The Gram matrix (similarities between training examples)
    /// - The dual coefficients learned during training
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = krr.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("kernelRidgeRegression.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize KernelRidgeRegression specific data
        writer.Write(Options.LambdaKRR);
        writer.Write((int)Options.DecompositionType);

        // Serialize _gramMatrix
        writer.Write(_gramMatrix.Rows);
        writer.Write(_gramMatrix.Columns);
        for (int i = 0; i < _gramMatrix.Rows; i++)
        {
            for (int j = 0; j < _gramMatrix.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_gramMatrix[i, j]));
            }
        }

        // Serialize _dualCoefficients
        writer.Write(_dualCoefficients.Length);
        for (int i = 0; i < _dualCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_dualCoefficients[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Kernel Ridge Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a Kernel Ridge Regression model from a byte array that was previously created using the
    /// Serialize method. It restores the base class data, model-specific options, the Gram matrix, and the dual
    /// coefficients, allowing the model to be used for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - All settings are restored
    /// - The Gram matrix is reconstructed
    /// - The dual coefficients are recovered
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("kernelRidgeRegression.model");
    /// 
    /// // Deserialize the model
    /// var options = new KernelRidgeRegressionOptions();
    /// var krr = new KernelRidgeRegression&lt;double&gt;(options);
    /// krr.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = krr.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize KernelRidgeRegression specific data
        Options.LambdaKRR = reader.ReadDouble();
        Options.DecompositionType = (MatrixDecompositionType)reader.ReadInt32();

        // Deserialize _gramMatrix
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _gramMatrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _gramMatrix[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize _dualCoefficients
        int length = reader.ReadInt32();
        _dualCoefficients = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            _dualCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Creates a new instance of the KernelRidgeRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new KernelRidgeRegression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the KernelRidgeRegression model with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new KernelRidgeRegression<T>((KernelRidgeRegressionOptions)Options, Regularization);
    }
}

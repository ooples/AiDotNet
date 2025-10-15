using System.Threading.Tasks;
using AiDotNet.Interpretability;

namespace AiDotNet.Regression;

/// <summary>
/// Base class for non-linear regression algorithms that provides common functionality for training and prediction.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements core functionality shared by different non-linear regression algorithms,
/// including kernel functions, regularization, and model serialization/deserialization.
/// </para>
/// <para>
/// Non-linear regression models can capture complex relationships in data that linear models cannot represent.
/// They typically use kernel functions to transform the input space into a higher-dimensional feature space
/// where the relationship becomes linear.
/// </para>
/// <para>
/// For Beginners:
/// Non-linear regression is used when your data doesn't follow a straight line pattern. These models can
/// capture curved or complex relationships between your input features and target values. Think of it like
/// having a flexible curve that can bend and shape itself to fit your data points, rather than just a
/// straight line.
/// </para>
/// </remarks>
public abstract class NonLinearRegressionModelBase<T> : INonLinearRegression<T>
{
    /// <summary>
    /// Gets the numeric operations provider for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for the numeric type T.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// This property provides a way to perform math operations (like addition, multiplication, etc.)
    /// on the generic type T. It allows the algorithm to work with different numeric types
    /// (float, double, decimal) without changing the core logic.
    /// </para>
    /// </remarks>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on the model parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks which input features have been manually
    /// selected as important for the non-linear regression model, regardless of what features
    /// are actually used in the model's calculations.
    /// 
    /// When set, these manually selected features take precedence over the automatic
    /// feature detection based on the model parameters.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// Gets the configuration options for the non-linear regression model.
    /// </summary>
    /// <value>
    /// Contains settings like kernel type, regularization parameters, and optimization parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// These are the settings that control how the model behaves. They include choices like what type of
    /// mathematical function (kernel) to use and how to prevent the model from memorizing the training data
    /// instead of learning general patterns (regularization).
    /// </para>
    /// </remarks>
    protected NonLinearRegressionOptions Options { get; private set; }

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An implementation of the IRegularization interface that determines how model complexity is penalized.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Regularization helps prevent "overfitting," which is when a model learns the training data too well
    /// and performs poorly on new data. It's like adding a penalty for complexity, encouraging the model
    /// to find simpler solutions that generalize better to new examples.
    /// </para>
    /// </remarks>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets or sets the support vectors used by the model.
    /// </summary>
    /// <value>
    /// A matrix where each row represents a support vector (a subset of the training examples that define the model).
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Support vectors are the key training examples that define the model's decision boundary.
    /// Instead of using all training data points, many non-linear models only need to remember
    /// a subset of them (the "support vectors") to make predictions. This makes the model more
    /// efficient and often helps it generalize better to new data.
    /// </para>
    /// </remarks>
    protected Matrix<T> SupportVectors { get; set; }

    /// <summary>
    /// Gets or sets the alpha coefficients for each support vector.
    /// </summary>
    /// <value>
    /// A vector of coefficients that determine the influence of each support vector on predictions.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Alpha coefficients determine how much influence each support vector has on the final prediction.
    /// Larger alpha values mean that support vector has a stronger effect on the prediction.
    /// </para>
    /// </remarks>
    protected Vector<T> Alphas { get; set; }

    /// <summary>
    /// Gets or sets the bias term (intercept) of the model.
    /// </summary>
    /// <value>
    /// A scalar value that represents the offset from the origin.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// The bias term (sometimes called the intercept) is like a baseline value for predictions.
    /// It shifts the entire model up or down to better fit the data. Without a bias term,
    /// all predictions would have to pass through the origin (0,0).
    /// </para>
    /// </remarks>
    protected T B { get; set; }

    /// <summary>
    /// Initializes a new instance of the NonLinearRegressionModelBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the non-linear regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with default values and prepares it for training.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the model with either the options you provide or default settings.
    /// It's like setting up a new tool before you start using it - you're configuring how it will work
    /// before you actually train it with data.
    /// </para>
    /// </remarks>
    protected NonLinearRegressionModelBase(NonLinearRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
    {
        Options = options ?? new NonLinearRegressionOptions();
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        SupportVectors = new Matrix<T>(0, 0);
        Alphas = new Vector<T>(0);
        B = NumOps.Zero;
    }

    /// <summary>
    /// Trains the non-linear regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Initializes the model parameters
    /// 3. Optimizes the model parameters using the training data
    /// 4. Extracts the support vectors and their coefficients
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. It looks at the examples you provide
    /// (input features and their corresponding target values) and adjusts its internal parameters to make
    /// predictions that match the target values as closely as possible. After training, the model will be
    /// ready to make predictions on new data.
    /// </para>
    /// </remarks>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        InitializeModel(x, y);
        OptimizeModel(x, y);
        ExtractModelParameters();
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the trained model to each input example and returns the predicted values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It takes your input features
    /// and runs them through the trained model to estimate what the target values should be. This is the
    /// main purpose of building a regression model - to predict values for new examples.
    /// </para>
    /// </remarks>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Validates the input data before training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in x doesn't match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method checks that the input data is valid before proceeding with training.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method makes sure your data is in the correct format before training begins.
    /// It checks that you have the same number of target values as you have examples in your input data.
    /// If not, it will raise an error to let you know there's a mismatch.
    /// </para>
    /// </remarks>
    protected virtual void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("The number of rows in X must match the length of y.");
        }
    }

    /// <summary>
    /// Initializes the model parameters before optimization.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the model parameters (Alphas and B) based on the input data.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Before the model can start learning, it needs to set up its internal values (parameters).
    /// This method creates those initial values, which will be adjusted during the training process.
    /// It's like setting up the starting position before beginning a journey.
    /// </para>
    /// </remarks>
    protected virtual void InitializeModel(Matrix<T> x, Vector<T> y)
    {
        // Initialize model parameters (e.g., Alphas, B) based on input data
        Alphas = new Vector<T>(x.Rows);
        B = NumOps.Zero;
    }

    /// <summary>
    /// Optimizes the model parameters using the training data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to perform the actual optimization
    /// of the model parameters.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This is where the actual learning happens. Different algorithms will implement this method
    /// differently, but they all have the same goal: to adjust the model's parameters to make
    /// predictions that match the training data as closely as possible, while still generalizing
    /// well to new data.
    /// </para>
    /// </remarks>
    protected abstract void OptimizeModel(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Extracts the support vectors and their coefficients after optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method identifies the support vectors (training examples with non-zero alpha coefficients)
    /// and extracts them along with their corresponding alpha values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, not all examples are equally important for making predictions.
    /// This method identifies which examples are most important (the "support vectors")
    /// and keeps only those for making future predictions. This makes the model more
    /// efficient and often improves its performance on new data.
    /// </para>
    /// </remarks>
    protected virtual void ExtractModelParameters()
    {
        // Extract support vectors and their corresponding alphas
        var supportVectorIndices = Enumerable.Range(0, Alphas.Length)
            .Where(i => NumOps.GreaterThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
            .ToArray();

        int featureCount = SupportVectors.Columns;
        SupportVectors = new Matrix<T>(supportVectorIndices.Length, featureCount);
        var newAlphas = new Vector<T>(supportVectorIndices.Length);

        for (int i = 0; i < supportVectorIndices.Length; i++)
        {
            int index = supportVectorIndices[i];
            for (int j = 0; j < featureCount; j++)
            {
                SupportVectors[i, j] = SupportVectors[index, j];
            }
            newAlphas[i] = Alphas[index];
        }

        Alphas = newAlphas;
    }

    /// <summary>
    /// Makes a prediction for a single input example.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the prediction for a single input example by applying the kernel function
    /// to the input and each support vector, multiplying by the corresponding alpha coefficient,
    /// and adding the bias term.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method calculates the prediction for a single example. It uses the support vectors
    /// (important training examples) and their weights (alpha coefficients) to compute how similar
    /// the new example is to each support vector. These similarity scores are combined to produce
    /// the final prediction. The mathematical function used to measure similarity is called the
    /// "kernel function."
    /// </para>
    /// </remarks>
    protected virtual T PredictSingle(Vector<T> input)
    {
        T result = B;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            Vector<T> supportVector = SupportVectors.GetRow(i);
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i], KernelFunction(input, supportVector)));
        }

        return result;
    }

        /// <summary>
    /// Computes the kernel function between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel function value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the kernel function value between two vectors based on the kernel type
    /// specified in the options. The kernel function measures the similarity between two vectors
    /// in a potentially higher-dimensional space.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The kernel function is a mathematical way to measure how similar two examples are to each other.
    /// Different kernel functions capture different types of similarities. For example:
    /// - Linear kernel: Measures similarity based on the dot product (like a straight-line relationship)
    /// - RBF (Radial Basis Function) kernel: Measures similarity based on distance (examples close to each other are more similar)
    /// - Polynomial kernel: Can capture more complex curved relationships
    /// 
    /// The kernel function is what gives non-linear regression models their power to model complex patterns.
    /// </para>
    /// </remarks>
    protected T KernelFunction(Vector<T> x1, Vector<T> x2)
    {
        switch (Options.KernelType)
        {
            case KernelType.Linear:
                return x1.DotProduct(x2);

            case KernelType.RBF:
                T squaredDistance = x1.Subtract(x2).Transform(v => NumOps.Square(v)).Sum();
                return NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-Options.Gamma), squaredDistance));

            case KernelType.Polynomial:
                T dot = x1.DotProduct(x2);
                return NumOps.Power(
                    NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Options.Gamma), dot), NumOps.FromDouble(Options.Coef0)),
                    NumOps.FromDouble(Options.PolynomialDegree)
                );

            case KernelType.Sigmoid:
                T sigmoidDot = x1.DotProduct(x2);
                return MathHelper.Tanh(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Options.Gamma), sigmoidDot), NumOps.FromDouble(Options.Coef0))
                );

            case KernelType.Laplacian:
                T l1Distance = x1.Subtract(x2).Transform(v => NumOps.Abs(v)).Sum();
                return NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-Options.Gamma), l1Distance));

            default:
                throw new NotImplementedException("Unsupported kernel type");
        }
    }

    /// <summary>
    /// Clips a value to be within the specified range.
    /// </summary>
    /// <param name="value">The value to clip.</param>
    /// <param name="low">The lower bound of the range.</param>
    /// <param name="high">The upper bound of the range.</param>
    /// <returns>The clipped value.</returns>
    /// <remarks>
    /// <para>
    /// This method ensures that a value is within the specified range by returning the lower bound
    /// if the value is less than the lower bound, the upper bound if the value is greater than the
    /// upper bound, or the value itself if it is within the range.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Clipping is a simple way to ensure a value stays within a certain range. It's like setting
    /// minimum and maximum limits. If the value is below the minimum, it's raised to the minimum.
    /// If it's above the maximum, it's lowered to the maximum. Otherwise, it stays as is.
    /// This is often used during optimization to keep parameters within valid bounds.
    /// </para>
    /// </remarks>
    protected T Clip(T value, T low, T high)
    {
        var max = NumOps.GreaterThan(value, low) ? value : low;
        return NumOps.LessThan(max, high) ? max : high;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and additional information
    /// such as the kernel type, kernel parameters, and the number of support vectors.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Model metadata is information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like what type of kernel it uses)
    /// and some statistics about its structure (like how many support vectors it has). This information
    /// can be useful for understanding the model's complexity and for debugging purposes.
    /// </para>
    /// </remarks>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
            {
                ["KernelType"] = Options.KernelType,
                ["Gamma"] = Options.Gamma,
                ["Coef0"] = Options.Coef0,
                ["PolynomialDegree"] = Options.PolynomialDegree,
                ["SupportVectorsCount"] = SupportVectors.Rows
            }
        };

        return metadata;
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to return the specific model type.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates what specific type of non-linear regression
    /// model this is (e.g., Support Vector Regression, Neural Network Regression, etc.). It's used
    /// internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected abstract ModelType GetModelType();

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including options, support vectors, alpha coefficients,
    /// bias term, and regularization options, to a byte array that can be stored or transmitted.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        var optionsJson = JsonConvert.SerializeObject(Options);
        writer.Write(optionsJson);

        // Serialize support vectors
        writer.Write(SupportVectors.Rows);
        writer.Write(SupportVectors.Columns);
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            for (int j = 0; j < SupportVectors.Columns; j++)
            {
                writer.Write(Convert.ToDouble(SupportVectors[i, j]));
            }
        }

        // Serialize alphas
        writer.Write(Alphas.Length);
        foreach (var alpha in Alphas)
        {
            writer.Write(Convert.ToDouble(alpha));
        }

        // Serialize B
        writer.Write(Convert.ToDouble(B));

        // Serialize regularization options
        var regularizationOptionsJson = JsonConvert.SerializeObject(Regularization.GetOptions());
        writer.Write(regularizationOptionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a new instance of the derived model class.
    /// </summary>
    /// <returns>A new instance of the same model type.</returns>
    /// <remarks>
    /// <para>
    /// This abstract factory method must be implemented by derived classes to create a new
    /// instance of their specific type. It's used by Clone and DeepCopy to ensure that
    /// the correct derived type is instantiated.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a new, empty instance of the specific model type.
    /// It's used during cloning and deep copying to ensure that the copy
    /// is of the same specific type as the original. This is more efficient
    /// than using reflection to create instances and gives derived classes
    /// explicit control over how new instances are created.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateInstance();

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including options,
    /// support vectors, alpha coefficients, bias term, and regularization options.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        var optionsJson = reader.ReadString();
        Options = JsonConvert.DeserializeObject<NonLinearRegressionOptions>(optionsJson) ?? new NonLinearRegressionOptions();

        // Deserialize support vectors
        int svRows = reader.ReadInt32();
        int svCols = reader.ReadInt32();
        SupportVectors = new Matrix<T>(svRows, svCols);
        for (int i = 0; i < svRows; i++)
        {
            for (int j = 0; j < svCols; j++)
            {
                SupportVectors[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize alphas
        int alphaCount = reader.ReadInt32();
        Alphas = new Vector<T>(alphaCount);
        for (int i = 0; i < alphaCount; i++)
        {
            Alphas[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize B
        B = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize regularization options
        var regularizationOptionsJson = reader.ReadString();
        var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson) 
            ?? new RegularizationOptions();

        // Create regularization based on deserialized options
        Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);

        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the model parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters (alpha coefficients and bias term).</returns>
    /// <remarks>
    /// <para>
    /// This method combines all model parameters into a single vector, with the bias term as the first element
    /// followed by all alpha coefficients. This representation is useful for optimization algorithms and
    /// for operations that need to treat all parameters uniformly.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method collects all the model's internal values (parameters) into a single list.
    /// Think of it like getting a complete list of ingredients and measurements for a recipe.
    /// This allows you to see all the parameters at once or pass them to other algorithms
    /// that work with the model's parameters as a group.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        // Create a vector to hold all parameters (bias + alphas)
        var parameters = new Vector<T>(Alphas.Length + 1);
    
        // Set the bias term as the first parameter
        parameters[0] = B;
    
        // Copy all alpha coefficients
        for (int i = 0; i < Alphas.Length; i++)
        {
            parameters[i + 1] = Alphas[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (bias term followed by alpha coefficients).</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when the parameters vector doesn't match the expected length.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance with the same structure as the current model but with different
    /// parameter values. The first element of the parameters vector is interpreted as the bias term, and the
    /// remaining elements are interpreted as alpha coefficients.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a new model with specific parameter values you provide.
    /// It's like following a recipe but changing the amounts of certain ingredients.
    /// This is useful when you want to experiment with different parameter settings
    /// or when an optimization algorithm suggests better parameter values.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        // Verify that the parameters vector has the correct length
        if (parameters.Length != Alphas.Length + 1)
        {
            throw new ArgumentException($"Parameters vector length ({parameters.Length}) " +
                                       $"does not match expected length ({Alphas.Length + 1}).");
        }
    
        // Create a new instance of the model
        var clone = (NonLinearRegressionModelBase<T>)this.Clone();
    
        // Set the bias term
        clone.B = parameters[0];
    
        // Set the alpha coefficients
        for (int i = 0; i < Alphas.Length; i++)
        {
            clone.Alphas[i] = parameters[i + 1];
        }
    
        return clone;
    }

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (bias term followed by alpha coefficients).</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the parameters vector doesn't match the expected length.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the model's parameters in place. The first element of the parameters vector
    /// is interpreted as the bias term, and the remaining elements are interpreted as alpha coefficients.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method updates the model's internal values (parameters) with new ones you provide.
    /// It's like adjusting the settings on a machine to change how it operates.
    /// This is useful when loading saved models or applying parameter updates
    /// from optimization algorithms.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // Verify that the parameters vector has the correct length
        if (parameters.Length != Alphas.Length + 1)
        {
            throw new ArgumentException($"Parameters vector length ({parameters.Length}) " +
                                       $"does not match expected length ({Alphas.Length + 1}).", nameof(parameters));
        }

        // Set the bias term
        B = parameters[0];

        // Set the alpha coefficients
        for (int i = 0; i < Alphas.Length; i++)
        {
            Alphas[i] = parameters[i + 1];
        }
    }

    /// <summary>
    /// Gets the indices of features that are actively used by the model.
    /// </summary>
    /// <returns>A collection of feature indices that have non-zero weight in the model.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies which features have a significant impact on the model's predictions by analyzing
    /// the support vectors and their coefficients. A feature is considered active if it has a non-zero weight
    /// in at least one support vector with a non-zero alpha coefficient.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method tells you which features (input variables) your model is actually using
    /// to make predictions. Some models may effectively ignore certain features if they
    /// don't help with predictions. Knowing which features are actually being used can help
    /// you understand what information the model considers important and potentially simplify
    /// your model by removing unused features.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // Create a set to store the active feature indices
        // This set will automatically remove duplicate indices and ensure that we only return unique values
        var activeIndices = new HashSet<int>(); activeIndices = new HashSet<int>();
    
        // Identify features that have non-zero weight in support vectors with non-zero alpha
        for (int i = 0; i < Alphas.Length; i++)
        {
            // Skip if the alpha coefficient is effectively zero
            if (NumOps.LessThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
                continue;
        
            // Check each feature in this support vector
            for (int j = 0; j < SupportVectors.Columns; j++)
            {
                // If the feature has a non-zero value, consider it active
                if (!NumOps.LessThan(NumOps.Abs(SupportVectors[i, j]), NumOps.FromDouble(1e-5)))
                {
                    activeIndices.Add(j);
                }
            }
        }
    
        return activeIndices;
    }

    /// <summary>
    /// Determines whether a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used by the model; otherwise, false.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when featureIndex is negative or greater than the number of features.</exception>
    /// <remarks>
    /// <para>
    /// This method checks whether a specific feature has a significant impact on the model's predictions
    /// by determining if it has a non-zero weight in at least one support vector with a non-zero alpha coefficient.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method checks if a specific feature (input variable) is actually being used
    /// by your model to make predictions. It's like checking if a particular ingredient
    /// in a recipe is actually affecting the final dish or if it could be left out without
    /// changing the result. This can help you understand what information your model
    /// considers important and potentially simplify your model by removing unused features.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        // Validate feature index
        if (featureIndex < 0 || featureIndex >= SupportVectors.Columns)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex), 
                $"Feature index must be between 0 and {SupportVectors.Columns - 1}.");
        }
    
        // Check if the feature has a non-zero value in any support vector with non-zero alpha
        for (int i = 0; i < Alphas.Length; i++)
        {
            // Skip if the alpha coefficient is effectively zero
            if (NumOps.LessThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
                continue;
        
            // Check if this feature has a non-zero value in this support vector
            if (!NumOps.LessThan(NumOps.Abs(SupportVectors[i, featureIndex]), NumOps.FromDouble(1e-5)))
            {
                return true;
            }
        }
    
        return false;
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    /// <returns>A new model instance that is a deep copy of the current model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the model, including all parameters,
    /// support vectors, and configuration options. Modifications to the returned model will not
    /// affect the original model, and vice versa.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a complete, independent copy of your model.
    /// It's like making a photocopy of a document - the copy looks exactly
    /// the same but is a separate object that can be modified without affecting
    /// the original. This is useful when you want to experiment with changes to
    /// a model without risking the original or when you need multiple independent
    /// instances of the same model (e.g., for ensemble learning).
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // Create a new instance through cloning
        var clone = (NonLinearRegressionModelBase<T>)this.Clone();
    
        // Perform deep copy of all mutable fields
        clone.SupportVectors = SupportVectors.Clone();
        clone.Alphas = Alphas.Clone();
        clone.B = B; // Value types are copied by value
        clone.Options = JsonConvert.DeserializeObject<NonLinearRegressionOptions>(
            JsonConvert.SerializeObject(Options)) ?? new NonLinearRegressionOptions();
    
        // Create a new regularization instance with the same options
        var regularizationOptions = Regularization.GetOptions();
        clone.Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);
    
        return clone;
    }

    /// <summary>
    /// Creates a shallow copy of the model.
    /// </summary>
    /// <returns>A new model instance that is a shallow copy of the current model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the model that shares references to the same internal data
    /// structures as the original model. This is primarily used internally by other methods that need
    /// to create modified copies of the model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a lightweight copy of your model.
    /// Unlike DeepCopy, which creates completely independent copies of everything,
    /// this method creates a new model object but may share some internal data
    /// with the original. This makes it faster to create copies, but changes to one
    /// copy might affect others in some cases. This is primarily used internally
    /// by other methods rather than directly by users.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        // Create a new instance using the factory method
        var clone = (NonLinearRegressionModelBase<T>)CreateInstance();
    
        // Copy the model parameters
        clone.SupportVectors = SupportVectors;  // Shallow copy
        clone.Alphas = Alphas;                 // Shallow copy
        clone.B = B;                          // Value types are copied by value
        clone.Options = Options;              // Shallow copy
        clone.Regularization = Regularization; // Shallow copy
    
        return clone;
    }

    /// <summary>
    /// Sets which features should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features should be considered active in the
    /// non-linear regression model, overriding the automatic determination based on the model parameters.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of whether they are used in model calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you manually tell the model which input features
    /// are important, regardless of what the model actually learned during training.
    /// 
    /// For example, if you have 10 features but want to focus on only features 2, 5, and 7,
    /// you can use this method to specify exactly those features. After setting these features:
    /// - Only these specific features will be reported as active by GetActiveFeatureIndices()
    /// - Only these features will return true when checked with IsFeatureUsed()
    /// - This selection will persist when the model is saved and loaded
    /// 
    /// This can be useful for:
    /// - Feature selection experiments (testing different feature subsets)
    /// - Simplifying model interpretation
    /// - Ensuring consistency across different models
    /// - Highlighting specific features you know are important from domain expertise
    /// </para>
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Initialize the hash set if it doesn't exist
        _explicitlySetActiveFeatures ??= [];

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} cannot be negative.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Matrix<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Matrix<T> input, Vector<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Matrix<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Matrix<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}
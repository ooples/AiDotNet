namespace AiDotNet.Regression;

public abstract class NonLinearRegressionBase<T> : INonLinearRegression<T>
{
    protected INumericOperations<T> NumOps { get; private set; }
    protected NonLinearRegressionOptions Options { get; private set; }
    protected IRegularization<T> Regularization { get; private set; }
    protected Matrix<T> SupportVectors { get; set; }
    protected Vector<T> Alphas { get; set; }
    protected T B { get; set; }

    protected NonLinearRegressionBase(NonLinearRegressionOptions? options = null, IRegularization<T>? regularization = null)
    {
        Options = options ?? new NonLinearRegressionOptions();
        Regularization = regularization ?? new NoRegularization<T>();
        NumOps = MathHelper.GetNumericOperations<T>();
        SupportVectors = new Matrix<T>(0, 0);
        Alphas = new Vector<T>(0);
        B = NumOps.Zero;
    }

    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        InitializeModel(x, y);
        OptimizeModel(x, y);
        ExtractModelParameters();
    }

    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    protected virtual void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("The number of rows in X must match the length of y.");
        }
    }

    protected virtual void InitializeModel(Matrix<T> x, Vector<T> y)
    {
        // Initialize model parameters (e.g., Alphas, B) based on input data
        Alphas = new Vector<T>(x.Rows);
        B = NumOps.Zero;
    }

    protected abstract void OptimizeModel(Matrix<T> x, Vector<T> y);

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

    protected T Clip(T value, T low, T high)
    {
        var max = NumOps.GreaterThan(value, low) ? value : low;
        return NumOps.LessThan(max, high) ? max : high;
    }

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

    protected abstract ModelType GetModelType();

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
        Regularization = RegularizationFactory.CreateRegularization<T>(regularizationOptions);

        NumOps = MathHelper.GetNumericOperations<T>();
    }
}
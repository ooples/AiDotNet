global using AiDotNet.DecompositionMethods;
global using AiDotNet.Factories;
global using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

public abstract class RegressionBase<T> : IRegression<T>
{
    protected INumericOperations<T> NumOps { get; private set; }
    protected RegressionOptions<T> Options { get; private set; }
    protected IRegularization<T> Regularization { get; private set; }

    public Vector<T> Coefficients { get; protected set; }
    public T Intercept { get; protected set; }

    public bool HasIntercept => Options.UseIntercept;

    protected RegressionBase(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        Regularization = regularization ?? new NoRegularization<T>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new RegressionOptions<T>();
        Coefficients = new Vector<T>(0);
        Intercept = NumOps.Zero;
    }

    public abstract void Train(Matrix<T> x, Vector<T> y);

    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = input.Multiply(Coefficients);

        if (Options.UseIntercept)
        {
            predictions = predictions.Add(Intercept);
        }

        return predictions;
    }

    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = Coefficients.Length,
            Complexity = Coefficients.Length,
            Description = $"{GetModelType()} model with {Coefficients.Length} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HasIntercept", HasIntercept },
                { "CoefficientNorm", Coefficients.Norm()! },
                { "FeatureImportances", CalculateFeatureImportances().ToArray() }
            }
        };
    }

    protected abstract ModelType GetModelType();

    protected virtual Vector<T> CalculateFeatureImportances()
    {
        return Coefficients.Transform(NumOps.Abs);
    }

    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "Coefficients", Coefficients.ToArray() },
            { "Intercept", Intercept ?? NumOps.Zero! },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataDict = JsonConvert.DeserializeObject<Dictionary<string, object>>(modelDataString);

        if (modelDataDict == null)
        {
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");
        }

        Coefficients = new Vector<T>((T[])modelDataDict["Coefficients"]);
        Intercept = (T)modelDataDict["Intercept"];

        var regularizationOptionsJson = JsonConvert.SerializeObject(modelDataDict["RegularizationOptions"]);
        var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson) 
            ?? throw new InvalidOperationException("Deserialization failed: Unable to deserialize regularization options.");
    
        Regularization = RegularizationFactory.CreateRegularization<T>(regularizationOptions);
    }

    protected Vector<T> SolveSystem(Matrix<T> a, Vector<T> b)
    {
        var decomposition = Options.DecompositionMethod;

        if (decomposition != null)
        {
            return decomposition.Solve(b);
        }
        else
        {
            // Use normal equation if specifically selected or as a fallback
            return SolveNormalEquation(a, b);
        }
    }

    private Vector<T> SolveNormalEquation(Matrix<T> a, Vector<T> b)
    {
        var aTa = a.Transpose().Multiply(a);
        var aTb = a.Transpose().Multiply(b);

        // Use LU decomposition for solving the normal equation
        var normalDecomposition = new NormalDecomposition<T>(aTa);
        return normalDecomposition.Solve(aTb);
    }
}
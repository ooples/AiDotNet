public class ErrorStats
{
    public double MAE { get; set; }  // Mean Absolute Error
    public double RMSE { get; set; } // Root Mean Squared Error
    public double MAD { get; set; }  // Mean Absolute Deviation
    public double MSE { get; set; }  // Mean Squared Error
    public double MAPE { get; set; } // Mean Absolute Percentage Error
    public double R2 { get; set; }   // R-squared (Coefficient of Determination)
    public double AdjustedR2 { get; set; }
    public double SampleStandardError { get; set; }
    public double PopulationStandardError { get; set; }
    public double AIC { get; set; }
    public double BIC { get; set; }
    public double AICAlt { get; set; }
    public double TestScore { get; set; }
    public double TestR2 { get; set; }
    public double TrainingScore { get; set; }
    public double TrainingR2 { get; set; }
    public double ValidationScore { get; set; }
    public double ValidationR2 { get; set; }
    public List<double> ErrorList { get; set; } = [];
}
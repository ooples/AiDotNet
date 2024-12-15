namespace AiDotNet.Models;

public class NormalizationParameters<T>
{
    private readonly INumericOperations<T> _numOps;

    public NormalizationParameters(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        Method = NormalizationMethod.None;
        Min = Max = Mean = StdDev = Scale = Shift = Median = IQR = P = _numOps.Zero;
        Bins = [];
    }

    public NormalizationMethod Method { get; set; }
    public T Min { get; set; }
    public T Max { get; set; }
    public T Mean { get; set; }
    public T StdDev { get; set; }
    public T Scale { get; set; }
    public T Shift { get; set; }
    public List<T> Bins { get; set; }
    public T Median { get; set; }
    public T IQR { get; set; }
    public T P { get; set; }
}
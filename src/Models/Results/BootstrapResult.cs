namespace AiDotNet.Models.Results
{
    public class BootstrapResult<T>
    {
        public T TrainingR2 { get; set; }
        public T ValidationR2 { get; set; }
        public T TestR2 { get; set; }

        public BootstrapResult()
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            TrainingR2 = numOps.Zero;
            ValidationR2 = numOps.Zero;
            TestR2 = numOps.Zero;
        }
    }
}
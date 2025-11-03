using Xunit;
using AiDotNet.MetaLearning;
using AiDotNet.LinearAlgebra;

namespace UnitTests.MetaLearning
{
    public class SEALModelTests
    {
        [Fact]
        public void Construct_And_Predict_Roundtrip_Succeeds()
        {
            var model = new SEALModel<double>();
            var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
            var output = model.Predict(input);
            Assert.Equal(input.Length, output.Length);
        }
    }
}

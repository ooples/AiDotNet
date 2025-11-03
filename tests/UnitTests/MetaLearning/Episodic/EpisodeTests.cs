using Xunit;
using AiDotNet.MetaLearning.Episodic;
using AiDotNet.LinearAlgebra;

namespace UnitTests.MetaLearning.Episodic
{
    public class EpisodeTests
    {
        [Fact]
        public void Episode_Constructs()
        {
            var sX = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 }));
            var sY = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 0, 1 }));
            var qX = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 3, 4 }));
            var qY = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 0 }));
            var ep = new Episode<double>(sX, sY, qX, qY);
            Assert.Equal(2, ep.SupportInputs.Length);
            Assert.Equal(2, ep.QueryLabels.Length);
        }
    }
}


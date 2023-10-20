using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AiDotNet.OutlierRemoval
{
    internal class MedianOutlierRemoval : IOutlierRemoval
    {
        internal override (double[] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[] rawInputs, double[] rawOutputs)
        {
            return (rawInputs, rawOutputs);
        }

        internal override (double[][] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[][] rawInputs, double[] rawOutputs)
        {
            return (rawInputs, rawOutputs);
        }
    }
}
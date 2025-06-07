using Bonsai;
using Bonsai.Design;
using Bonsai.Vision.Design;
using System;

[assembly: TypeVisualizer(typeof(FacialLandmarks.FacialAnalysisOutputVisualizer),
    Target = typeof(FacialLandmarks.FacialAnalysisOutput))]

namespace FacialLandmarks
{
    public class FacialAnalysisOutputVisualizer : DialogTypeVisualizer
    {
        public override void Load(IServiceProvider serviceProvider)
        {
        }

        public override void Show(object value)
        {
        }

        public override void Unload()
        {
        }
    }
}
using Bonsai;
using Bonsai.Design;
using Bonsai.Vision.Design;
using OpenCV.Net;
using System;
using System.Linq;

[assembly: TypeVisualizer(typeof(FacialLandmarks.FacialAnalysisOutputImageOverlay),
    Target = typeof(MashupSource<ImageMashupVisualizer, FacialLandmarks.FacialAnalysisOutputVisualizer>))]

namespace FacialLandmarks
{
    public class FacialAnalysisOutputImageOverlay : DialogTypeVisualizer
    {
        private ImageMashupVisualizer visualizer;

        public override void Load(IServiceProvider provider)
        {
            visualizer = (ImageMashupVisualizer)provider.GetService(typeof(MashupVisualizer));
        }

        public override void Show(object value)
        {
            var image = visualizer.VisualizerImage;
            var facialAnalysis = (FacialAnalysisOutput)value;

            foreach (var face in facialAnalysis.Faces)
            {
                CV.Rectangle(
                    image,
                    new Point(face.X, face.Y),
                    new Point(face.X + face.Width, face.Y + face.Height),
                    new Scalar(0, 255, 0, 1),
                    2,
                    LineFlags.AntiAliased
                );
            }

            foreach (var facialLandmarks in facialAnalysis.Landmarks)
            {
                foreach (var landmark in facialLandmarks.Landmarks)
                {
                    CV.Circle(
                        image,
                        new Point(landmark.X, landmark.Y),
                        2,
                        new Scalar(255, 0, 0, 1),
                        -1,
                        LineFlags.AntiAliased
                    );
                }
            }
        }

        public override void Unload()
        {
            visualizer = null;
        }
    }
}
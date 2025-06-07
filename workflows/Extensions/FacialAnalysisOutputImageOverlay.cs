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
        private IplImage overlay;

        public override void Load(IServiceProvider provider)
        {
            visualizer = (ImageMashupVisualizer)provider.GetService(typeof(MashupVisualizer));
        }

        public override void Show(object value)
        {
            var image = visualizer.VisualizerImage;
            var facialAnalysis = (FacialAnalysisOutput)value;

            for (int i = 0; i < facialAnalysis.FaceCount; i++)
            {
                var face = facialAnalysis.Faces[i];
                var landmarks = facialAnalysis.Landmarks[i].Landmarks;

                CV.Rectangle(
                    image,
                    new Point(face.X, face.Y),
                    new Point(face.X + face.Width, face.Y + face.Height),
                    new Scalar(0, 255, 0, 1),
                    2,
                    LineFlags.AntiAliased
                );

                foreach (var landmark in landmarks)
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

            // // Iterate over each detected face.
            // for (int i = 0; i < facialAnalysis.FaceCount; i++)
            // {
            //     var face = facialAnalysis.Faces[i];
            //     var landmarks = facialAnalysis.Landmarks[i].Landmarks;

            //     // Draw the bounding box for the face.
            //     CV.Rectangle(
            //         image,
            //         new Point(face.X, face.Y),
            //         new Point(face.X + face.Width, face.Y + face.Height),
            //         new Scalar(0, 255, 0, 1),
            //         2,
            //         LineFlags.AntiAliased
            //     );

            //     // Draw face mesh: iterate each triangle defined in the connectivity list.
            //     foreach (var tri in FaceMeshTriangles)
            //     {
            //         // Make sure that the triangle indices are within bounds.
            //         if (tri[0] < landmarks.Length && tri[1] < landmarks.Length && tri[2] < landmarks.Length)
            //         {
            //             var p1 = new Point(landmarks[tri[0]].X, landmarks[tri[0]].Y);
            //             var p2 = new Point(landmarks[tri[1]].X, landmarks[tri[1]].Y);
            //             var p3 = new Point(landmarks[tri[2]].X, landmarks[tri[2]].Y);

            //             // Draw triangle edges.
            //             CV.Line(image, p1, p2, new Scalar(0, 0, 255, 1), 1, LineFlags.AntiAliased);
            //             CV.Line(image, p2, p3, new Scalar(0, 0, 255, 1), 1, LineFlags.AntiAliased);
            //             CV.Line(image, p3, p1, new Scalar(0, 0, 255, 1), 1, LineFlags.AntiAliased);
            //         }
            //     }
            // }

        }

        public override void Unload()
        {
            if (overlay != null)
            {
                overlay.Dispose();
            }
            visualizer = null;
        }
    }
}
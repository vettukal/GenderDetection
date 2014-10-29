using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Xml;
using Emgu.CV;
using Emgu.CV.GPU;
using Emgu.CV.Structure;
using Emgu.Util;

namespace RGBD_Gender_Detector
{
    class Program
    {
        String readXmlNode(String XMLPath, String nodeName)
        {
            XmlDocument doc = new XmlDocument();
            doc.Load(XMLPath);
            XmlNode node = doc.DocumentElement.SelectSingleNode("/data/" + nodeName);
            return node.InnerText;
        }

        int getGrayScalePixelLBP(int[,] mask)
        {
            int[,] lbpMask = new int[3, 3];
            int i = 0, j = 0;
            int[] lbpIndexI = new int[] { 0, 0, 0, 1, 2, 2, 2, 1 };
            int[] lbpIndexJ = new int[] { 0, 1, 2, 2, 2, 1, 0, 0 };
            int[] lbpFeature = new int[8];
            int featureValue = 0;
            while (i < lbpIndexI.Length)
            {
                if (mask[lbpIndexI[i], lbpIndexJ[j]] >= mask[1, 1])
                    lbpFeature[i] = 1;
                else
                    lbpFeature[i] = 0;
                i++;
                j++;
            }

            for (int p = 0; p < 8; p++)
            {
                featureValue += (int)(lbpFeature[p] * Math.Pow(2, p));
            }
            return featureValue;
        }

        int[,] getGrayScaleImageLBP(Image<Gray, byte> grayScaleImage)
        {
            int[,] mask = new int[3, 3];

            int[,] grayScaleLBP = new int[grayScaleImage.Width, grayScaleImage.Height];

            for (int i = 0; i < grayScaleImage.Height; i++)
            {
                for (int j = 0; j < grayScaleImage.Width; j++)
                {
                    if (j - 1 < 0 || i - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[0, 0] = grayScaleImage.Data[j - 1, i - 1, 0];

                    if (i - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[0, 1] = grayScaleImage.Data[j, i - 1, 0];

                    if (j + 1 > grayScaleImage.Width || i - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[0, 2] = grayScaleImage.Data[j + 1, i - 1, 0];

                    if (j - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[1, 0] = grayScaleImage.Data[j - 1, i, 0];

                    mask[1, 1] = grayScaleImage.Data[j, i, 0];

                    if (j + 1 > grayScaleImage.Width) mask[0, 0] = 0;
                    else
                        mask[1, 2] = grayScaleImage.Data[j + 1, i, 0];

                    if (i + 1 > grayScaleImage.Height || j - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[2, 0] = grayScaleImage.Data[j - 1, i + 1, 0];

                    if (i + 1 > grayScaleImage.Height) mask[0, 0] = 0;
                    else
                        mask[2, 1] = grayScaleImage.Data[j, i + 1, 0];

                    if (i + 1 > grayScaleImage.Height || j + 1 > grayScaleImage.Width) mask[0, 0] = 0;
                    else
                        mask[2, 2] = grayScaleImage.Data[j + 1, i + 1, 0];

                    grayScaleLBP[j, i] = getGrayScalePixelLBP(mask);

                }
            }


            return grayScaleLBP;

        }

        int[,] getUniformLBP(int[,] lbpFeature, int height, int width)
        {
            int[,] ulbpFeature = new int[width,height];

            int[] lookUp = new int[] { 0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 
                58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
                58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
                58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
                24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 
                32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
                58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 
                58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 
                44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 
                58, 53, 54, 55, 56, 57 };

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    ulbpFeature[j, i] = lookUp[lbpFeature[j, i]];

                }
            }

            return ulbpFeature;
        }


        long[] getU2LBPHistogramVector(int[,] grayScaleLbp, int height, int width)
        {
            long[] pixelDataArray = new long[59];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    pixelDataArray[grayScaleLbp[j, i]]++;

                }
            }

            return pixelDataArray;
        }

        List<long[]> getGrayScaleImageU2LBPFeatureVector(String grayScaleImagePath)
        {
            Image<Gray, byte> grayScaleImage = new Image<Gray, Byte>(grayScaleImagePath);
            int subImageHeight = grayScaleImage.Height / 8;
            int subImageWidth = grayScaleImage.Width / 8;
            Image<Gray, byte> uLBPImage = grayScaleImage = new Image<Gray, Byte>(grayScaleImagePath);
            List<long[]> uLBPfeatureVector = new List<long[]>();
            List<long[]> gLBPfeatureVector = new List<long[]>();
            int[,] subImageVector;
            Image<Gray, Byte> segment, segmentHolder;


            int i_, j_ ,i ,j;
            for (j = 0, j_ = 0; j < grayScaleImage.Height - subImageHeight; j += subImageHeight, j_++)
            {
                for (i = 0, i_ = 0; i < grayScaleImage.Width - subImageHeight; i += subImageWidth, i_++)
                {
                    segmentHolder = grayScaleImage.Clone();
                    segmentHolder.ROI = new Rectangle(i, j, subImageWidth, subImageHeight);
                    segment = segmentHolder.Clone();
                    subImageVector = getUniformLBP(getGrayScaleImageLBP(segment), segment.Height, segment.Width);
                    uLBPfeatureVector.Add(getU2LBPHistogramVector(subImageVector, segment.Height, segment.Width));
                    
                }
            }

            return uLBPfeatureVector;
        }


        List<long[]> getDepthImageGLBPFeatureVector(String depthImagePath)
        {
            Image<Gray, byte> grayScaleImage = new Image<Gray, Byte>(depthImagePath);
            int subImageHeight = grayScaleImage.Height / 8;
            int subImageWidth = grayScaleImage.Width / 8;
            Image<Gray, byte> uLBPImage = grayScaleImage = new Image<Gray, Byte>(depthImagePath);
            List<long[]> uLBPfeatureVector = new List<long[]>();
            List<long[]> gLBPfeatureVector = new List<long[]>();
            int[,] subImageVector;
            Image<Gray, Byte> segment, segmentHolder;


            int i_, j_, i, j;
            for (j = 0, j_ = 0; j < grayScaleImage.Height - subImageHeight; j += subImageHeight, j_++)
            {
                for (i = 0, i_ = 0; i < grayScaleImage.Width - subImageHeight; i += subImageWidth, i_++)
                {
                    segmentHolder = grayScaleImage.Clone();
                    segmentHolder.ROI = new Rectangle(i, j, subImageWidth, subImageHeight);
                    segment = segmentHolder.Clone();
                    subImageVector = getUniformLBP(getGrayScaleImageLBP(segment), segment.Height, segment.Width);
                    uLBPfeatureVector.Add(getU2LBPHistogramVector(subImageVector, segment.Height, segment.Width));

                }
            }

            return uLBPfeatureVector;
        }


        static void Main(string[] args)
        {
            Program program = new Program();
            String configXML = "config.xml";
            String testingDataNode = "testing_data";
            String trainingDataNode = "training_data";
            FileStream BGRfileStream, DepthfileStream;
            String imagePath, depthImageDir, BGRImagePath;
            Console.WriteLine(program.readXmlNode(configXML, testingDataNode));
            int j = 0;
            String[] depthImageFilePaths, BGRImageFilePaths;
            for (int i = 1; i <= 2; i++)
            {
                imagePath = program.readXmlNode(configXML, trainingDataNode) + "\\" + i;
                depthImageDir = imagePath + "\\Depth\\";
                BGRImagePath = imagePath + "\\RGB\\";
                depthImageFilePaths = Directory.GetFiles(@depthImageDir);
                BGRImageFilePaths = Directory.GetFiles(@BGRImagePath);
                j = 0;
                while (j < depthImageFilePaths.Length)
                {
                    Console.WriteLine(depthImageFilePaths[j]);
                    Console.WriteLine(BGRImageFilePaths[j]);
                    program.getGrayScaleImageU2LBPFeatureVector(BGRImageFilePaths[j]);
                    program.getDepthImageGLBPFeatureVector(depthImageFilePaths[j]);
                    Console.WriteLine();
                    j++;
                }
                try
                {
                    // read from file or write to file
                }
                finally
                {
                    ;
                    //fileStream.Close();
                }
            }

        }
    }
}

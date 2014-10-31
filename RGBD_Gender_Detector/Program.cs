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

        int max(int a, int b)
        {
            if (a >= b)
                return a;
            else
                return b;
        }

        int min(int a, int b)
        {
            if (a < b)
                return a;
            else
                return b;
        }

        int[] getDepthPixelGLBP(int[,] mask)
        {
            int[,] lbpMask = new int[3, 3];
            int i = 0, j = 0;
            int[] lbpIndexI = new int[] { 2, 2, 2, 1 };
            int[] lbpIndexJ = new int[] { 2, 1, 0, 0 };
            int[] glbpFeature = new int[4];
            while (i < lbpIndexI.Length)
            {
                glbpFeature[i] = max(min(mask[lbpIndexI[i], lbpIndexJ[j]] - mask[1, 1], 7), -8);
                i++;
                j++;
            }

            return glbpFeature;
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


        List<int[,]> getDepthImageGLBP(Image<Gray, byte> depthImage)
        {
            int[,] mask = new int[3, 3];

            List<int[,]> depthGLBPList = new List<int[,]>();
            int[,] depthGLBP1 = new int[depthImage.Width, depthImage.Height];
            int[,] depthGLBP2 = new int[depthImage.Width, depthImage.Height];
            int[,] depthGLBP3 = new int[depthImage.Width, depthImage.Height];
            int[,] depthGLBP4 = new int[depthImage.Width, depthImage.Height];
            int[] pixelGlbp = new int[4];

            for (int i = 0; i < depthImage.Height; i++)
            {
                for (int j = 0; j < depthImage.Width; j++)
                {
                    if (j - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[1, 0] = depthImage.Data[j - 1, i, 0];

                    mask[1, 1] = depthImage.Data[j, i, 0];

                    if (i + 1 > depthImage.Height || j - 1 < 0) mask[0, 0] = 0;
                    else
                        mask[2, 0] = depthImage.Data[j - 1, i + 1, 0];

                    if (i + 1 > depthImage.Height) mask[0, 0] = 0;
                    else
                        mask[2, 1] = depthImage.Data[j, i + 1, 0];

                    if (i + 1 > depthImage.Height || j + 1 > depthImage.Width) mask[0, 0] = 0;
                    else
                        mask[2, 2] = depthImage.Data[j + 1, i + 1, 0];


                    pixelGlbp = getDepthPixelGLBP(mask);

                    depthGLBP1[j, i] = pixelGlbp[0];
                    depthGLBP2[j, i] = pixelGlbp[1];
                    depthGLBP3[j, i] = pixelGlbp[2];
                    depthGLBP4[j, i] = pixelGlbp[3];

                }
            }
            depthGLBPList.Add(depthGLBP1);
            depthGLBPList.Add(depthGLBP2);
            depthGLBPList.Add(depthGLBP3);
            depthGLBPList.Add(depthGLBP4);

            return depthGLBPList;

        }

        int[,] getUniformLBP(int[,] lbpFeature, int height, int width)
        {
            int[,] ulbpFeature = new int[width, height];

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


        long[] getGLBPHistogramVector(int[,] depthGlbp, int height, int width)
        {
            long[] pixelDataArray = new long[16];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    pixelDataArray[depthGlbp[j, i] + 8]++;

                }
            }

            return pixelDataArray;
        }

        long[] getConcatenatedFeatureVector(List<long[]> featureVectorList)
        {
            long[] concatenatedVector;
            int vectorSize = 0, j_ = 0;
            for (int i = 0; i < featureVectorList.Count; i++)
            {
                vectorSize += featureVectorList[i].Length;
            }

            concatenatedVector = new long[vectorSize];

            for (int i = 0; i < featureVectorList.Count; i++)
            {
                for (int j = 0; j < featureVectorList[i].Length; j++, j_++)
                {
                    concatenatedVector[j_] = featureVectorList[i][j];
                }

            }


            return concatenatedVector;

        }

        List<long[]> getGrayScaleImageU2LBPFeatureVector(String grayScaleImagePath)
        {
            Image<Gray, byte> grayScaleImage = new Image<Gray, Byte>(grayScaleImagePath);
            int subImageHeight = grayScaleImage.Height / 1 - 1;
            int subImageWidth = grayScaleImage.Width / 1 - 1;
            Image<Gray, byte> uLBPImage = grayScaleImage;
            List<long[]> uLBPfeatureVector = new List<long[]>();
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


        List<long[]> getDepthImageGLBPFeatureVector(String depthImagePath)
        {
            Image<Gray, byte> depthImage = new Image<Gray, Byte>(depthImagePath);
            int subImageHeight = depthImage.Height / 1 - 1;
            int subImageWidth = depthImage.Width / 1 - 1;
            Image<Gray, byte> uLBPImage = depthImage;
            List<long[]> gLBPConcatenatedfeatureVector = new List<long[]>();
            List<long[]> gLBPfeatureOrientationVectorList = null;
            List<int[,]> depthGLBPList = new List<int[,]>();
            Image<Gray, Byte> segment, segmentHolder;


            int i_, j_, i, j;
            for (j = 0, j_ = 0; j < depthImage.Height - subImageHeight; j += subImageHeight, j_++)
            {
                for (i = 0, i_ = 0; i < depthImage.Width - subImageHeight; i += subImageWidth, i_++)
                {
                    segmentHolder = depthImage.Clone();
                    segmentHolder.ROI = new Rectangle(i, j, subImageWidth, subImageHeight);
                    segment = segmentHolder.Clone();
                    depthGLBPList = getDepthImageGLBP(segment);
                    gLBPfeatureOrientationVectorList = new List<long[]>();
                    for (int k = 0; k < 4; k++)
                        gLBPfeatureOrientationVectorList.Add(getGLBPHistogramVector(depthGLBPList[k], segment.Height, segment.Width));

                    gLBPConcatenatedfeatureVector.Add(getConcatenatedFeatureVector(gLBPfeatureOrientationVectorList));
                    gLBPfeatureOrientationVectorList = null;

                }
            }

            return gLBPConcatenatedfeatureVector;
        }

        private int[,] getGLBP(int[,] p1, int p2, int p3)
        {
            throw new NotImplementedException();
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
            System.IO.StreamWriter file = new System.IO.StreamWriter("d:\\dataset.txt");
            try
            {
                for (int i = 1; i <= 20; i++)
                {
                    imagePath = program.readXmlNode(configXML, trainingDataNode) + "\\" + i;
                    depthImageDir = imagePath + "\\Depth\\";
                    BGRImagePath = imagePath + "\\RGB\\";
                    depthImageFilePaths = Directory.GetFiles(@depthImageDir);
                    BGRImageFilePaths = Directory.GetFiles(@BGRImagePath);
                    List<long[]> u2LBPfvList = new List<long[]>();
                    List<long[]> gLBPfvList = new List<long[]>();
                    List<long[]> combinedFVlist = null;
                    
                    

                    j = 0;

                    while (j < depthImageFilePaths.Length)
                    {
                        Console.WriteLine(depthImageFilePaths[j]);
                        Console.WriteLine(BGRImageFilePaths[j]);
                        u2LBPfvList = program.getGrayScaleImageU2LBPFeatureVector(BGRImageFilePaths[j]);
                        gLBPfvList = program.getDepthImageGLBPFeatureVector(depthImageFilePaths[j]);

                        string genderCode = System.IO.File.ReadAllText(@imagePath + "\\gender.txt");

                        
                        String featureVector = genderCode;

                        //Here range of k depends on the number of parts of the image that has been made
                        // so for 8 x 8 parts of image k value ranges from 0 to 63
                        
                        combinedFVlist = new List<long[]>();

                        for (int k = 0; k < 1; k++)
                        {
                            combinedFVlist.Add(program.getConcatenatedFeatureVector(new List<long[]> { u2LBPfvList[k], gLBPfvList[k] }));
                        }
                        long[] ImageFeatureVector = program.getConcatenatedFeatureVector(combinedFVlist);

                        combinedFVlist = null;

                        for (int l = 0; l < ImageFeatureVector.Length; l++)
                        {
                            int index = l + 1;
                            featureVector += " " + index + ":" + ImageFeatureVector[l];
                        }


                        file.WriteLine(featureVector);
                        file.WriteLine();

                        featureVector = null;

                        Console.WriteLine();
                        j++;
                    }

                    // read from file or write to file

                }
            }
            finally
            {
                file.Close();

                //fileStream.Close();
            }

        }
    }
}

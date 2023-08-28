/* Vehicle Controller (Agent)
 * @Author: Tarun Gupta (tarung@smu.edu.sg)
 * Singapore Management University
 */

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Configurer : MonoBehaviour
{
    public static string baseName = "5x5_test";
    //public static string basePath = @"/Users/tarun/Desktop/Unity Environment/Assets/ML-Agents/Examples/MAPF/";
    public static string basePath = @"C:\Users\jjling.2018\ml-agents\UnitySDK\Assets\ML-Agents\Examples\MAPF\";
    public static string modelsPath = basePath + @"Models\";
    public static string configPath = basePath + @"Configs\";
    public static string modelFileNameNetworkX = modelsPath + baseName + ".model";
    public static string gridObst = modelsPath + baseName + ".txt";
    public static string modelFileNameInfo = modelsPath + baseName + ".info";
    public static string configFileName = configPath + baseName + ".config";
    public static string separateRangesFile = modelsPath + baseName + "-INPUT.model";
}

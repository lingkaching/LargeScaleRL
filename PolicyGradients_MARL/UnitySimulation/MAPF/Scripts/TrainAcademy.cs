/* Train Academy
 * @Author: Tarun Gupta (tarung@smu.edu.sg)
 * Singapore Management University
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System.IO;
using System.Linq;

public class TrainAcademy : Academy, IComparer<GameObject>
{
    private GameObject[] allIZones;
    private GameObject[] allNormalZones;
    private int agentsToSpawn;
    public GameObject vehiclePrefab;
    public Brain vehicleBrain;
    private static List<GameObject> allZones;
    private static GameObject[] allVehicles;
    private readonly float EPSILON = 1e-5f;

    private bool writeModel;
    private bool viewIDs;
    private bool readConfig;

    private enum AgentSpawning {Random, Subset, Fixed};
    private int agentSpawningMethod = (int)AgentSpawning.Random;

    private enum ZoneTransitionGen {Random, Common, Separate};
    private int zoneTransitionGenMethod = (int)ZoneTransitionGen.Random;
    private int minCommnonZoneTransitionValue;
    private int maxCommnonZoneTransitionValue;
    private int minCapacityValue;
    private int maxCapacityValue;

    private List<int> GoalsPerAgent;
    private List<GameObject> possibleSpawnZones;
    private List<GameObject> possibleGoalZones;

    private List<List<int> > fixedAgentSpec;

    private List<List<int>> separateZoneTranSpec;

    public static GameObject GetZoneByID(int id)
    {
        if (id >= 0 && id < allZones.Count)
        {
            return allZones[id];
        }
        else
        {
            Debug.LogWarning("Invalid Zone ID -- Doesn't exist.");
            return null;
        }
    }

    private void SetDefaultConfiguration()
    {
        this.GetComponent<Academy>().resetParameters.TryGetValue("AgentsToSpawn", out float agents_count);
        agentsToSpawn = (int)agents_count;
        agentSpawningMethod = (int)AgentSpawning.Random;
        zoneTransitionGenMethod = (int)ZoneTransitionGen.Random;
        readConfig = false;
        possibleGoalZones.Clear();
        possibleSpawnZones.Clear();
        GoalsPerAgent.Clear();
        fixedAgentSpec.Clear();
        fixedAgentSpec = null;
        separateZoneTranSpec.Clear();
        separateZoneTranSpec = null;
        possibleSpawnZones.AddRange(allIZones);
        possibleGoalZones.AddRange(allIZones);
        GoalsPerAgent = Enumerable.Repeat(1, agentsToSpawn).ToList();
        minCommnonZoneTransitionValue = 1;
        maxCommnonZoneTransitionValue = 20;
        minCapacityValue = 2;
        maxCapacityValue = 2;
    }

    private List<int> ReadObstacleFile()
    {
        if (!File.Exists(Configurer.gridObst))
        {
            throw new System.Exception("Input Data file " + Configurer.gridObst + " does not exist.");
        }
        using (StreamReader sr = new StreamReader(Configurer.gridObst))
        {
            string line;
            int obst;
            List<int> obstList = new List<int>();          
            while ((line = sr.ReadLine()) != null)
            {
                if (line.Contains("###"))
                {
                    continue;
                }
                obst = int.Parse(line);
                obstList.Add(obst);
            }
            return obstList;
        }
    }

    public override void InitializeAcademy()
    {
        allIZones = GameObject.FindGameObjectsWithTag("IZone");
        allNormalZones = GameObject.FindGameObjectsWithTag("Zone");
        allZones = new List<GameObject>();
        if(allIZones.Length != 0) 
        {
            allZones.AddRange(allIZones);
        }
        if(allNormalZones.Length != 0) 
        {
            allZones.AddRange(allNormalZones);
        }
        possibleGoalZones = new List<GameObject>();
        possibleSpawnZones = new List<GameObject>();
        GoalsPerAgent = new List<int>();
        fixedAgentSpec = new List<List<int>>();
        separateZoneTranSpec = new List<List<int>>();
        minCommnonZoneTransitionValue = 1;
        maxCommnonZoneTransitionValue = 20;
        minCapacityValue = 2;
        maxCapacityValue = 2;

        allZones.Sort(Compare);

        List<Vector3> allZoneCoords = new List<Vector3>();
        List<float> distancesToCheck = new List<float>();
        distancesToCheck.Add(0.5f);
        distancesToCheck.Add(1f);

        //10x10
        // var landmarkList = new List<int> { 28, 83, 19, 43, 26};
        //5x5
        var landmarkList = new List<int> { 7,14,17,6,9};

        for (int i = 0; i < allZones.Count; i++)
        {
            allZones[i].GetComponent<ZoneController>().SetZoneID(i);
            allZoneCoords.Add(allZones[i].transform.position);
            if (landmarkList.Contains(i))
            {
                allZones[i].GetComponent<Renderer>().material.color = new Color32(255,255,40,50);
            }
        }

        // Uncomment this for loop for grid based maps! And comment it for other ones if you want to manually specify the neighbours yourselves!
        for (int i = 0; i < allZones.Count; i++)
        {
            List<int> obstList = ReadObstacleFile();
            if (obstList.Contains(i))
            {
                continue;
            }

            GameObject[] existing_neighbours = allZones[i].GetComponent<ZoneController>().GetNeighbourZones();
            List<GameObject> newNeighs = new List<GameObject>();
            newNeighs.AddRange(existing_neighbours);
            
            // AddToNeighbours(allZoneCoords, allZones[i].transform.position, newNeighs);

            float dist_1;
            if (allZones[i].CompareTag("IZone"))
            {
                dist_1 = 0.5f;
            }
            else
            {
                dist_1 = 1f;
            }

            Vector3 target;

            foreach (float dist_2 in distancesToCheck)
            {
                // +x direction check
                target = allZones[i].transform.position;
                target.x += (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);

                // -x direction check
                target = allZones[i].transform.position;
                target.x -= (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);

                // +y direction check
                target = allZones[i].transform.position;
                target.y += (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);

                // -y direction check
                target = allZones[i].transform.position;
                target.y -= (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);

                // +z direction check
                target = allZones[i].transform.position;
                target.z += (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);

                // -z direction check
                target = allZones[i].transform.position;
                target.z -= (dist_1 + dist_2);
                AddToNeighbours(allZoneCoords, target, newNeighs, obstList);
            }

            Debug.Log("Length of neighbours of zone " + allZones[i].GetComponent<ZoneController>().GetZoneID() + " changed from " + existing_neighbours.Length + " to " + newNeighs.Count);
            allZones[i].GetComponent<ZoneController>().SetNeighbourZones(newNeighs.ToArray());
        }
        Debug.Log("Total Number of Zones: " + allZones.Count);
    }

    private bool AddToNeighbours(List<Vector3> allZoneCoords, Vector3 target, List<GameObject> newNeighs, List<int> obstList)
    {
        int result = allZoneCoords.IndexOf(target);
        if (result != -1)
        {
            if (newNeighs.IndexOf(allZones[result]) == -1)
            {
                if (!obstList.Contains(result))
                {
                    newNeighs.Add(allZones[result]);                   
                }
                return true;               
            }
        }
        return false;
    }
    
    public int Compare(GameObject x, GameObject y)
    {
        Vector3 xPos = x.transform.position;
        Vector3 yPos = y.transform.position;
        if (xPos.x != yPos.x)
        {
            return xPos.x.CompareTo(yPos.x);
        }
        else if (xPos.y != yPos.y)
        {
            return xPos.y.CompareTo(yPos.y);
        }
        else
        {
            return xPos.z.CompareTo(yPos.z);
        }
    }
    
    private bool ReadConfigurationFile()
    {
        string filename = Configurer.configFileName;
        if (!File.Exists(Configurer.modelFileNameNetworkX))
        {
            Debug.LogWarning("Please generate model first by running WriteModel#0F#1T = 1.0 and visualize the model using ViewIDs#0F#1T = 1.0 before using the config file.");
        }

        try
        {
            // Create an instance of StreamReader to read from a file.
            // The using statement also closes the StreamReader.
            using (StreamReader sr = new StreamReader(filename))
            {
                string line;
                // Read and display lines from the file until 
                // the end of the file is reached. 
                while ((line = sr.ReadLine()) != null)
                {
                    if (line.Contains("##"))
                    {
                        continue;
                    }
                    if (line.Contains("="))
                    {
                        string[] tokens = line.Split('=');
                        switch (tokens[0].Trim())
                        {
                            case "agentsToSpawn":
                                agentsToSpawn = int.Parse(tokens[1].Trim());
                                break;
                            case "agentSpawning":
                                switch (tokens[1].Trim())
                                {
                                    case "random":
                                        agentSpawningMethod = (int)AgentSpawning.Random;
                                        break;
                                    case "subset":
                                        agentSpawningMethod = (int)AgentSpawning.Subset;
                                        break;
                                    case "fixed":
                                        agentSpawningMethod = (int)AgentSpawning.Fixed;
                                        break;
                                    default:
                                        agentSpawningMethod = (int)AgentSpawning.Random;
                                        break;
                                }
                                break;
                            case "random:possibleGoalZones":
                                if (agentSpawningMethod == (int)AgentSpawning.Random)
                                {
                                    switch (tokens[1].Trim())
                                    {
                                        case "All":
                                            possibleGoalZones.Clear();
                                            possibleGoalZones.AddRange(allZones);
                                            break;
                                        case "IZones":
                                            possibleGoalZones.Clear();
                                            possibleGoalZones.AddRange(allIZones);
                                            break;
                                    }
                                }
                                break;
                            case "random:possibleSpawnZones":
                                if (agentSpawningMethod == (int)AgentSpawning.Random)
                                {
                                    switch (tokens[1].Trim())
                                    {
                                        case "All":
                                            possibleSpawnZones.Clear();
                                            possibleSpawnZones.AddRange(allZones);
                                            break;
                                        case "IZones":
                                            possibleSpawnZones.Clear();
                                            possibleSpawnZones.AddRange(allIZones);
                                            break;
                                    }
                                }
                                break;
                            case "subset:possibleSpawnZones":
                                if (agentSpawningMethod == (int)AgentSpawning.Subset)
                                {
                                    if(tokens[1].Trim().Contains(","))
                                    {
                                        string[] splitStringValues = tokens[1].Trim().Split(',');
                                        if (splitStringValues.Length > 0)
                                        {
                                            possibleSpawnZones.Clear();
                                            for(int i=0; i<splitStringValues.Length; i++)
                                            {
                                                int givenZoneID = int.Parse(splitStringValues[i].Trim());
                                                GameObject tryGetZoneByID = GetZoneByID(givenZoneID);
                                                if (tryGetZoneByID != null)
                                                {
                                                    possibleSpawnZones.Add(tryGetZoneByID);
                                                }
                                                else
                                                {
                                                    throw new System.Exception("Invalid zone ID in subset:possibleSpawnZones in config file: " + givenZoneID);
                                                }
                                            }
                                            if (possibleSpawnZones.Count != splitStringValues.Length)
                                            {
                                                throw new System.Exception("All values in subset:possibleSpawnZones couldn't be parsed properly.");
                                            }
                                        }
                                        else
                                        {
                                            throw new System.Exception("Invalid length of subset:possibleSpawnZones in config file.");
                                        }
                                    }
                                    else
                                    {
                                        int givenZoneID = int.Parse(tokens[1].Trim());
                                        GameObject tryGetZoneByID = GetZoneByID(givenZoneID);
                                        if (tryGetZoneByID != null)
                                        {
                                            possibleSpawnZones.Add(tryGetZoneByID);
                                        }
                                        else
                                        {
                                            throw new System.Exception("Invalid zone ID in subset:possibleSpawnZones in config file: " + givenZoneID);
                                        }
                                    }
                                }
                                break;
                            case "subset:possibleGoalZones":
                                if (agentSpawningMethod == (int)AgentSpawning.Subset)
                                {
                                    if (tokens[1].Trim().Contains(","))
                                    {
                                        string[] splitStringValues = tokens[1].Trim().Split(',');
                                        if (splitStringValues.Length > 0)
                                        {
                                            possibleGoalZones.Clear();
                                            for (int i = 0; i < splitStringValues.Length; i++)
                                            {
                                                int givenZoneID = int.Parse(splitStringValues[i].Trim());
                                                GameObject tryGetZoneByID = GetZoneByID(givenZoneID);
                                                if (tryGetZoneByID != null)
                                                {
                                                    possibleGoalZones.Add(tryGetZoneByID);
                                                }
                                                else
                                                {
                                                    throw new System.Exception("Invalid zone ID in subset:possibleGoalZones in config file: " + givenZoneID);
                                                }
                                            }
                                            if (possibleGoalZones.Count != splitStringValues.Length)
                                            {
                                                throw new System.Exception("All values in subset:possibleGoalZones couldn't be parsed properly.");
                                            }
                                        }
                                        else
                                        {
                                            throw new System.Exception("Invalid length of subset:possibleGoalZones in config file.");
                                        }
                                    }
                                    else
                                    {
                                        int givenZoneID = int.Parse(tokens[1].Trim());
                                        GameObject tryGetZoneByID = GetZoneByID(givenZoneID);
                                        if (tryGetZoneByID != null)
                                        {
                                            possibleGoalZones.Add(tryGetZoneByID);
                                        }
                                        else
                                        {
                                            throw new System.Exception("Invalid zone ID in subset:possibleGoalZones in config file: " + givenZoneID);
                                        }
                                    }
                                }
                                break;
                            case "goalsPerAgent":
                                if (agentSpawningMethod != (int)AgentSpawning.Fixed)
                                {
                                    if (tokens[1].Trim().Contains(","))
                                    {
                                        string[] listOfGoals = tokens[1].Trim().Split(',');
                                        if (listOfGoals.Length != agentsToSpawn)
                                        {
                                            throw new System.Exception("Error: Invalid Length of list goalsPerAgent in config file");
                                        }
                                        GoalsPerAgent.Clear();
                                        for (int i = 0; i < listOfGoals.Length; i++)
                                        {
                                            GoalsPerAgent.Add(int.Parse(listOfGoals[i].Trim()));
                                            if (GoalsPerAgent[i] > possibleGoalZones.Count)
                                            {
                                                throw new System.Exception("Error: Number of goals specified is more than length of possible set of goal zones.");
                                            }
                                        }
                                    }
                                    else
                                    {
                                        GoalsPerAgent = Enumerable.Repeat(int.Parse(tokens[1].Trim()), agentsToSpawn).ToList();
                                    }
                                }
                                break;
                            case "fixed:AgentSpawning":
                                if (agentSpawningMethod == (int)AgentSpawning.Fixed)
                                {
                                    fixedAgentSpec.Clear();
                                    for (int i = 0; i < agentsToSpawn; i++)
                                    {
                                        fixedAgentSpec.Add(new List<int>());
                                    }
                                    if (tokens[1].Trim().Contains(";"))
                                    {
                                        string[] splitTokens = tokens[1].Trim().Split(';');
                                        // Debug.Log("Hooha1 " + System.String.Join("; ", splitTokens));
                                        if (splitTokens.Length != agentsToSpawn)
                                        {
                                            throw new System.Exception("Invalid length of list fixed:AgentSpawning in config file. Should be same as number of agents.");
                                        }
                                        for (int i = 0; i < splitTokens.Length; i++)
                                        {
                                            if (splitTokens[i].Trim().Contains(","))
                                            {
                                                string[] moreSplit = splitTokens[i].Trim().Split(',');
                                                // Debug.Log("Hooha2 " + System.String.Join("; ", moreSplit));

                                                if (moreSplit.Length < 3)
                                                {
                                                    throw new System.Exception("Invalid list fixed:AgentSpawning in config file. Need atleast 3 argument (id, spawn, goal(s)) for each agent.");
                                                }
                                                int agentID = int.Parse(moreSplit[0].Trim());
                                                int spawnZoneID = int.Parse(moreSplit[1].Trim());
                                                if(agentID >= agentsToSpawn || agentID < 0)
                                                {
                                                    throw new System.Exception("fixed:AgentSpawning Invalid agent ID : " + agentID);
                                                }
                                                GameObject spawnZone = GetZoneByID(spawnZoneID);
                                                if (spawnZone == null)
                                                {
                                                    throw new System.Exception("fixed:AgentSpawning Invalid spawn Zone ID : " + spawnZoneID);
                                                }
                                                fixedAgentSpec[agentID].Add(spawnZoneID);
                                                // Debug.Log("Hooha3 agent " + agentID +" : "  + System.String.Join("; ", fixedAgentSpec[agentID]));


                                                int[] goalZoneIDs = new int[moreSplit.Length - 2];
                                                for (int j=0; j<goalZoneIDs.Length; j++)
                                                {
                                                    goalZoneIDs[j] = int.Parse(moreSplit[j+2].Trim());
                                                    int ID = goalZoneIDs[j];
                                                    GameObject gZone = GetZoneByID(ID);
                                                    if (gZone == null)
                                                    {
                                                        throw new System.Exception("fixed:AgentSpawning Invalid goal Zone ID : " + ID);
                                                    }
                                                    if (ID == spawnZoneID)
                                                    {
                                                        throw new System.Exception("fixed:AgentSpawning - One of the goal zones " + ID + " is same as spawn zones. ");
                                                    }
                                                    if (fixedAgentSpec[agentID].Contains(ID))
                                                    {
                                                        throw new System.Exception("fixed:AgentSpawning - One of the goal zones " + ID + " is same as another goal zones. ");
                                                    }
                                                    fixedAgentSpec[agentID].Add(ID);
                                                    // Debug.Log("Hooha3 agent " + agentID + " : " + System.String.Join("; ", fixedAgentSpec[agentID]));

                                                }
                                            }
                                            else
                                            {
                                                throw new System.Exception("Invalid list fixed:AgentSpawning in config file. Please use comma separation.");
                                            }
                                        }
                                        for(int i=0; i<agentsToSpawn; i++)
                                        {
                                            if (fixedAgentSpec[i].Count == 0)
                                            {
                                                throw new System.Exception("You may have used an ID more than once in fixed:AgentSpawning. Data for agent " + i +" missing.");
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if (tokens[1].Trim().Contains(","))
                                        {
                                            string[] moreSplit = tokens[1].Trim().Split(',');
                                            if (moreSplit.Length < 3)
                                            {
                                                throw new System.Exception("Invalid list fixed:AgentSpawning in config file. Need atleast 3 argument (id, spawn, goal(s)) for each agent.");
                                            }
                                            int agentID = int.Parse(moreSplit[0].Trim());
                                            int spawnZoneID = int.Parse(moreSplit[1].Trim());
                                            if (agentID >= agentsToSpawn || agentID < 0)
                                            {
                                                throw new System.Exception("fixed:AgentSpawning Invalid agent ID : " + agentID);
                                            }

                                            GameObject spawnZone = GetZoneByID(spawnZoneID);
                                            if (spawnZone == null)
                                            {
                                                throw new System.Exception("fixed:AgentSpawning Invalid spawn Zone ID : " + spawnZoneID);
                                            }
                                            fixedAgentSpec[agentID].Add(spawnZoneID);

                                            int[] goalZoneIDs = new int[moreSplit.Length - 2];
                                            for (int j = 0; j < goalZoneIDs.Length; j++)
                                            {
                                                goalZoneIDs[j] = int.Parse(moreSplit[j + 2].Trim());
                                                int ID = goalZoneIDs[j];
                                                GameObject gZone = GetZoneByID(ID);
                                                if (gZone == null)
                                                {
                                                    throw new System.Exception("fixed:AgentSpawning - Invalid goal Zone ID : " + ID);
                                                }
                                                if (ID == spawnZoneID)
                                                {
                                                    throw new System.Exception("fixed:AgentSpawning - One of the goal zones " + ID +  " is same as spawn zones. ");
                                                }
                                                if (fixedAgentSpec[agentID].Contains(ID))
                                                {
                                                    throw new System.Exception("fixed:AgentSpawning - One of the goal zones " + ID + " is same as another goal zones. ");
                                                }
                                                fixedAgentSpec[agentID].Add(ID);
                                            }
                                        }
                                        else
                                        {
                                            throw new System.Exception("Invalid list fixed:AgentSpawning in config file. Please use comma separation.");
                                        }
                                    }
                                }
                                break;
                            case "zoneTransitionAndCapacityGen":
                                switch (tokens[1].Trim())
                                {
                                    case "random":
                                        zoneTransitionGenMethod = (int)ZoneTransitionGen.Random;
                                        break;
                                    case "commonRanges":
                                        zoneTransitionGenMethod = (int)ZoneTransitionGen.Common;
                                        break;
                                    case "separateRanges":
                                        zoneTransitionGenMethod = (int)ZoneTransitionGen.Separate;
                                        if (!File.Exists(Configurer.modelFileNameNetworkX))
                                        {
                                            throw new System.Exception("separateRanges:values - Please generate model file first before using separate Ranges.");
                                        }
                                        ReadSeparateRangesFile();
                                        break;
                                    default:
                                        zoneTransitionGenMethod = (int)ZoneTransitionGen.Random;
                                        break;
                                }
                                break;
                            case "commonRanges:values":
                                if(zoneTransitionGenMethod == (int)ZoneTransitionGen.Common)
                                {
                                    if (tokens[1].Trim().Contains(","))
                                    {
                                        string[] splittedTokens = tokens[1].Trim().Split(',');
                                        if (splittedTokens.Length != 4)
                                        {
                                            throw new System.Exception("commonRanges:values - Require 4 values for tmin, tmax, mincapacity and maxcapacity.");
                                        }
                                        minCommnonZoneTransitionValue = int.Parse(splittedTokens[0].Trim());
                                        maxCommnonZoneTransitionValue = int.Parse(splittedTokens[1].Trim());
                                        minCapacityValue = int.Parse(splittedTokens[2].Trim());
                                        maxCapacityValue = int.Parse(splittedTokens[3].Trim());
                                        if (maxCommnonZoneTransitionValue < minCommnonZoneTransitionValue || maxCapacityValue < minCapacityValue)
                                        {
                                            throw new System.Exception("commonRanges:values - maxValue needs to be >= minValue.");
                                        }
                                        if(minCommnonZoneTransitionValue < 1 || minCapacityValue < 1)
                                        {
                                            throw new System.Exception("commonRanges:values - minValue cannot be less than 1.");
                                        }
                                    }
                                    else
                                    {
                                        throw new System.Exception("commonRanges:values - Use comma as delimiter.");
                                    }
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
            }
            return true;
        }
        catch (System.Exception e)
        {
            // This catch block is used to make sure that we can continue to random initializations in case of parsing failure.
            //Debug.LogWarning(e.Message);
            //Debug.LogWarning("File Could Not Be Read, Switching to Random Initializations.");
            throw new System.Exception(e.Message);
            //return false; If you want to continue to random initializations, please comment the exception in above line and uncomment this.
        }
    }

    private void ReadSeparateRangesFile()
    {
        if (!File.Exists(Configurer.separateRangesFile))
        {
            throw new System.Exception("Input Data file " + Configurer.separateRangesFile + " does not exist.");
        }
        using (StreamReader sr = new StreamReader(Configurer.separateRangesFile))
        {
            separateZoneTranSpec.Clear();
            for(int i=0; i<allZones.Count; i++)
            {
                separateZoneTranSpec.Add(new List<int>());
            }

            string line;
            // Read and display lines from the file until 
            // the end of the file is reached. 
            while ((line = sr.ReadLine()) != null)
            {
                if (line.Contains("##") || line.Trim().Length == 0)
                {
                    continue;
                }

                string[] tokens = line.Trim().Split(' ');
                if (tokens.Length < 3)
                {
                    throw new System.Exception("Too few arguments in input data file. Need zone, capacity, and neighbour tuples.");
                }

                int zone = int.Parse(tokens[0].Trim());
                GameObject actualZone = GetZoneByID(zone);
                if(actualZone == null)
                {
                    throw new System.Exception("Invalid zone ID in input data file: " + zone);
                }

                if(separateZoneTranSpec[zone].Count != 0)
                {
                    throw new System.Exception("You might have added data for zone: " + zone + " more than once.");
                }

                separateZoneTranSpec[zone].Add(zone);

                int capZone = int.Parse(tokens[1].Trim());
                if (capZone < 1)
                {
                    throw new System.Exception("Invalid capacity of zone " + zone + " in input data file: " + capZone);
                }

                separateZoneTranSpec[zone].Add(capZone);

                GameObject[] correctNeighbours = actualZone.GetComponent<ZoneController>().GetNeighbourZones();
                int numberOfNeighbours = tokens.Length - 2;
                for(int i=0; i<numberOfNeighbours; i++)
                {
                    string neighbourData = tokens[i + 2];
                    if (!neighbourData.Contains(','))
                    {
                        throw new System.Exception("Use comma separation for neighbour data of zone " +zone);
                    }
                    string[] splitTokens = tokens[i + 2].Trim().Split(',');
                    if(splitTokens.Length != 3)
                    {
                        throw new System.Exception("Need atleast three components (neighbour id, t_min, t_max) for neighbour data of zone " + zone);
                    }
                    int neighbourID = int.Parse(splitTokens[0].Trim());
                    GameObject actualNeighbour = GetZoneByID(neighbourID);
                    if (actualNeighbour == null)
                    {
                        throw new System.Exception("Invalid neighbour ID in input data file for zone " + zone + " neighbour " + neighbourID);
                    }

                    if (!correctNeighbours.Contains(actualNeighbour))
                    {
                        throw new System.Exception("Neighbour ID " + neighbourID +" is not a neighbour of zone " + zone);
                    }

                    int t_min_neighbour = int.Parse(splitTokens[1].Trim());
                    int t_max_neighbour = int.Parse(splitTokens[2].Trim());
                    if(t_min_neighbour > t_max_neighbour)
                    {
                        throw new System.Exception("Min value greater than max for zone " + zone + " neighbour " +neighbourID);
                    }

                    separateZoneTranSpec[zone].Add(neighbourID);
                    separateZoneTranSpec[zone].Add(t_min_neighbour);
                    separateZoneTranSpec[zone].Add(t_max_neighbour);
                }
            }
            for (int i = 0; i < separateZoneTranSpec.Count; i++)
            {
                if (separateZoneTranSpec[i].Count == 0)
                {
                    throw new System.Exception("You may have added data for an ID more than once in input data file. Data for zone " + i + " missing.");
                }

                GameObject[] neighbours = allZones[i].GetComponent<ZoneController>().GetNeighbourZones();
                var extractedNeighbours = separateZoneTranSpec[i].Skip(2).Where((x, j) => j % 3 == 0);
                //Debug.Log(string.Join(", ", extractedNeighbours));
                foreach (GameObject neighbour in neighbours)
                {
                    int nid = neighbour.GetComponent<ZoneController>().GetZoneID();
                    if (!extractedNeighbours.Contains(nid))
                    {
                        throw new System.Exception("You have missed the data for neighbour " + nid + " of zone " + i);
                    }
                }
            }
        }
    }

    public bool AlmostEquals(float double1, float double2)
    {
        return (Mathf.Abs(double1 - double2) <= EPSILON);
    }

    public bool AlmostEquals(Vector3 vec1, Vector3 vec2)
    {
        return (Mathf.Abs(vec1.x - vec2.x) <= EPSILON) &&
               (Mathf.Abs(vec1.y - vec2.y) <= EPSILON) &&
               (Mathf.Abs(vec1.z - vec2.z) <= EPSILON);
    }

    private Vector3 SpawnAdjustmentForNormalZones(GameObject spawnZone)
    {
        if (spawnZone.CompareTag("Zone"))
        {
            float adjustment = 0.99f;
            GameObject[] neighs = spawnZone.GetComponent<ZoneController>().GetNeighbourZones();
            if (neighs.Length == 1)
            {
                Vector3 neighbourPosition = neighs[0].gameObject.transform.position;
                Vector3 spawnPos = spawnZone.transform.position;
                Vector3 diff_vector = neighbourPosition - spawnPos;

                //Debug.Log(" -- X " + diff_vector.x);
                //Debug.Log(" -- Y " + diff_vector.y);
                //Debug.Log(" -- Z " + diff_vector.z);

                if (!AlmostEquals(diff_vector.x, 0.0f) && AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.z, 0.0f))
                {
                    Debug.Log(this.GetStepCount() + " -- X-Condition True");
                    if (diff_vector.x.CompareTo(0.0f) < 0)
                    {
                        Debug.Log(this.GetStepCount() + " -- X-Condition True +1");
                        spawnPos.x += adjustment;
                    }
                    else
                    {
                        Debug.Log(this.GetStepCount() + " -- X-Condition True -1");
                        spawnPos.x -= adjustment;
                    }
                }
                else if (!AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.x, 0.0f) && AlmostEquals(diff_vector.z, 0.0f))
                {
                    Debug.Log(this.GetStepCount() + " -- Y-Condition True");
                    if (diff_vector.y.CompareTo(0.0f) < 0)
                    {
                        Debug.Log(this.GetStepCount() + " -- Y-Condition True +1");
                        spawnPos.y += adjustment;
                    }
                    else
                    {
                        Debug.Log(this.GetStepCount() + " -- Y-Condition True -1");
                        spawnPos.y -= adjustment;
                    }
                }
                else if (!AlmostEquals(diff_vector.z, 0.0f) && AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.x, 0.0f))
                {
                    Debug.Log(this.GetStepCount() + " -- Z-Condition True");
                    if (diff_vector.z.CompareTo(0.0f) < 0)
                    {
                        Debug.Log(this.GetStepCount() + " -- Z-Condition True +1");
                        spawnPos.z += adjustment;
                    }
                    else
                    {
                        Debug.Log(this.GetStepCount() + " -- Z-Condition True -1");
                        spawnPos.z -= adjustment;
                    }
                }
                else
                {
                    Debug.LogWarning(this.GetStepCount() + " -- NO - Condition True");
                }
                return spawnPos;
            }
            else
            {
                Debug.Log("Normal zones with more than one neighbour should not be there.");
            }
        }
        return new Vector3(-1f, -1f, -1f);
    }

    private void RandomAndSubsetSpawning()
    {
        for (int i = 0; i < agentsToSpawn; i++)
        {
            int goalsThisAgent = GoalsPerAgent[i];

            // Number of goals for the agent.
            Debug.Log("Number of Goals for Agent " + i + " : " + goalsThisAgent);

            // Spawning a new UAV vehicle agent
            var select_random_spawn_zone = Random.Range(0, possibleSpawnZones.Count);
            GameObject spawnZone = possibleSpawnZones[select_random_spawn_zone];
            Vector3 spawnPosition = spawnZone.transform.position;
            if (spawnZone.CompareTag("Zone"))
            {
                spawnPosition = SpawnAdjustmentForNormalZones(spawnZone);
            }

            GameObject newVehicle = Instantiate(vehiclePrefab, spawnPosition, Quaternion.identity) as GameObject;
            newVehicle.name = "Vehicle - " + i;
            Debug.Log("Spawned Agent " + i + " : " + spawnZone.GetComponent<ZoneController>().GetZoneID());

            // Assigning a learning brain to the UAV agent.            
            // newVehicle.GetComponent<Agent>().GiveBrain(vehicleBrain);

            GameObject[] goalsForVehicle = new GameObject[goalsThisAgent];
            List<GameObject> availableGoalZones = new List<GameObject>();
            availableGoalZones.AddRange(possibleGoalZones);
            if (availableGoalZones.Contains(spawnZone))
            {
                availableGoalZones.Remove(spawnZone);
            }
            // Assigning a goal to the UAV agent
            for (int j = 0; j < goalsThisAgent; j++)
            {
                var select_random_goal_zone = Random.Range(0, availableGoalZones.Count);
                goalsForVehicle[j] = availableGoalZones[select_random_goal_zone];
                Debug.Log("Agent " + i + " Goal " + j + " : " + goalsForVehicle[j].GetComponent<ZoneController>().GetZoneID());
                availableGoalZones.RemoveAt(select_random_goal_zone);
            }
            newVehicle.GetComponent<VehicleController>().SetGoalZones(goalsForVehicle);
        }

        // Assigning IDs to vehicles and getting access to all vehicle objects.
        allVehicles = GameObject.FindGameObjectsWithTag("Vehicle");
        for (int i = 0; i < allVehicles.Length; i++)
        {
            allVehicles[i].GetComponent<VehicleController>().SetVehicleID(i);
        }
    }

    // Will only be successfully called if readConfig=True and there is no parsing error in reading config file and fixed sampling is mentioned in config file.
    private void FixedAgentSpawning()
    {
        if (fixedAgentSpec == null || fixedAgentSpec.Count != agentsToSpawn)
        {
            Debug.LogWarning("Unexpected Error. Parsing of config file failed.");
        }

        allVehicles = new GameObject[agentsToSpawn];
        for (int i = 0; i < agentsToSpawn; i++)
        {

            // Spawning a new UAV vehicle agent
            var selected_spawn_zone = fixedAgentSpec[i][0];
            GameObject spawnZone = GetZoneByID(selected_spawn_zone);
            Vector3 spawnPosition = spawnZone.transform.position;
            if (spawnZone.CompareTag("Zone"))
            {
                spawnPosition = SpawnAdjustmentForNormalZones(spawnZone);
            }

            GameObject newVehicle = Instantiate(vehiclePrefab, spawnPosition, Quaternion.identity) as GameObject;
            newVehicle.name = "Vehicle - " + i;
            Debug.Log("Spawned Agent " + i + " : " + spawnZone.GetComponent<ZoneController>().GetZoneID());

            int goalsThisAgent = fixedAgentSpec[i].Count - 1;
            GameObject[] goalsForVehicle = new GameObject[goalsThisAgent];
            for (int j = 0; j < goalsThisAgent; j++)
            {
                var selected_goal_zone = fixedAgentSpec[i][j + 1];
                goalsForVehicle[j] = GetZoneByID(selected_goal_zone);
                Debug.Log("Agent " + i + " Goal " + j + " : " + goalsForVehicle[j].GetComponent<ZoneController>().GetZoneID());
            }
            newVehicle.GetComponent<VehicleController>().SetGoalZones(goalsForVehicle);
            newVehicle.GetComponent<VehicleController>().SetVehicleID(i);
            allVehicles[i] = newVehicle;
        }
    }

    private void ZoneTransitionGeneration()
    {
        if (zoneTransitionGenMethod != (int)ZoneTransitionGen.Separate)
        {
            foreach (GameObject obj in allZones)
            {
                GameObject[] neighbours = obj.GetComponent<ZoneController>().GetNeighbourZones();
                List<List<int>> transitionArray = new List<List<int>>();            
                for (int i=0; i<neighbours.Length; i++)
                {
                    List<int> listForNeighbour = new List<int>
                    {
                        neighbours[i].GetComponent<ZoneController>().GetZoneID(),
                        minCommnonZoneTransitionValue,
                        maxCommnonZoneTransitionValue
                    };
                    transitionArray.Add(listForNeighbour);
                    Debug.Log("Start Zone " + obj.GetComponent<ZoneController>().GetZoneID() + " Neighbour Zone " + transitionArray[i][0] + " Transition Time " + transitionArray[i][1] + " " + transitionArray[i][2]);
                }
                obj.GetComponent<ZoneController>().SetNeighbourTransitions(transitionArray);
                obj.GetComponent<ZoneController>().SetCapacity(Random.Range(minCapacityValue, maxCapacityValue+1));
                Debug.Log("Capacity for zone " + obj.GetComponent<ZoneController>().GetZoneID() + " : "  + obj.GetComponent<ZoneController>().GetCapacity());
            }
        }
        else
        {
            foreach (List<int> zoneInfo in separateZoneTranSpec)
            {
                int zone = zoneInfo[0];
                GameObject obj = GetZoneByID(zone);
                obj.GetComponent<ZoneController>().SetCapacity(zoneInfo[1]);
                Debug.Log("Set capacity of zone " + zone + " as : " + obj.GetComponent<ZoneController>().GetCapacity());

                List<List<int>> transitionArray = new List<List<int>>();
                for (int i = 2; i < zoneInfo.Count; i += 3)
                {
                    int first_neighbour_id = zoneInfo[i];
                    int tmin = zoneInfo[i + 1];
                    int tmax = zoneInfo[i + 2];
                    List<int> listForNeighbour = new List<int>
                    {
                        first_neighbour_id,
                        tmin,
                        tmax
                    };
                    transitionArray.Add(listForNeighbour);
                    Debug.Log("Start Zone " + obj.GetComponent<ZoneController>().GetZoneID() + " Neighbour Zone " + listForNeighbour[0] + " Transition Time " + listForNeighbour[1] + " " + listForNeighbour[2]);
                }
                obj.GetComponent<ZoneController>().SetNeighbourTransitions(transitionArray);
            }
        }
    }

    private void DestroyExistingVehiclesForReset()
    {
        allVehicles = GameObject.FindGameObjectsWithTag("Vehicle");
        if (allVehicles.Length > 0)
        {
            for (int i = 0; i < allVehicles.Length; i++)
            {
                Destroy(allVehicles[i].gameObject);
            }
        }
    }

    private void ClearZoneCountForReset()
    {
        allIZones = GameObject.FindGameObjectsWithTag("IZone");
        // allNormalZones = GameObject.FindGameObjectsWithTag("Zone");
        if (allIZones.Length > 0)
        {
            for (int i = 0; i < allIZones.Length; i++)
            {
                allIZones[i].GetComponent<ZoneController>().currentAgents = new List<GameObject>();
            }
        }
    }

    public override void AcademyReset()
    {
        Debug.Log("Academy Reset Called");

        this.GetComponent<Academy>().resetParameters.TryGetValue("WriteModel#0F#1T", out float s_write_model);
        this.GetComponent<Academy>().resetParameters.TryGetValue("ReadConfig#0F#1T", out float s_read_config);
        this.GetComponent<Academy>().resetParameters.TryGetValue("ViewIDs#0F#1T", out float s_view_ids);

        writeModel = (System.Math.Abs(s_write_model - 1.0f) < EPSILON);
        readConfig = (System.Math.Abs(s_read_config - 1.0f) < EPSILON);
        viewIDs = (System.Math.Abs(s_view_ids - 1.0f) < EPSILON);

        bool is_read_successful = false;
        if (readConfig)
        {
            Debug.Log("Reading from Configuration File");
            is_read_successful = ReadConfigurationFile();
            Debug.Log("File Correctly Read Status: " + is_read_successful);
        }

        if (!is_read_successful || !readConfig)
        {
            Debug.Log("Resorting to default Configuration");
            SetDefaultConfiguration();
        }

        ZoneTransitionGeneration();

        if (writeModel)
        {
            string filename = Configurer.modelFileNameInfo;
            string anotherFile = Configurer.modelFileNameNetworkX;
            if (File.Exists(filename))
            {
                Debug.LogWarning("File " + filename + " Exists, It will be overwritten. If you want to override, remove the file and re-run again.");
            }

            try
            {
                Debug.Log("Writing model information to file");
                StreamWriter sw = new StreamWriter(filename);
                StreamWriter swAnother = new StreamWriter(anotherFile);
                sw.WriteLine("## Count of all zones: " + allZones.Count);
                sw.WriteLine("## Count of IZones: " + allIZones.Length);
                sw.WriteLine("## Count of Normal Zones: " + allNormalZones.Length);
                swAnother.WriteLine("## Count of all zones: " + allZones.Count);
                swAnother.WriteLine("## Count of IZones: " + allIZones.Length);
                swAnother.WriteLine("## Count of Normal Zones: " + allNormalZones.Length);

                if (sw != null && swAnother != null)
                {
                    for (int i = 0; i < allZones.Count; i++)
                    {
                        if (allZones[i].CompareTag("IZone"))
                        {
                            sw.Write("I-" + i + " ");
                        }
                        else
                        {
                            sw.Write(i + " ");
                        }

                        sw.Write(allZones[i].GetComponent<ZoneController>().GetCapacity() + " ");
                        swAnother.Write(i + " ");

                        foreach (GameObject obj in allZones[i].GetComponent<ZoneController>().GetNeighbourZones())
                        {
                            if (obj.CompareTag("IZone"))
                            {
                                sw.Write("I-");
                            }
                            sw.Write("(" + obj.GetComponent<ZoneController>().GetZoneID() + ",");

                            List<int> tInfo = allZones[i].GetComponent<ZoneController>().GetTransitionInfo(obj);
                            if (tInfo != null)
                            {
                                sw.Write(tInfo[0] + "," + tInfo[1] + ") ");
                            }
                            else
                            {
                                Debug.LogWarning("Should not happen.");
                                sw.Write("null ");
                            }
                            swAnother.Write(obj.GetComponent<ZoneController>().GetZoneID() + " ");
                        }
                        sw.Write("\n");
                        swAnother.Write("\n");
                    }
                    sw.Close();
                    swAnother.Close();
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning(e.Message);
                Debug.Log(e.StackTrace);
                Debug.LogWarning("Model Files Could Not Be Written.");
            }
        }

        if (viewIDs)
        {
            if (!GetIsInference())
            {
                Debug.LogWarning("System is running in train mode, will not be able to visualize anything. Run with train_mode = False.");
            }
            else
            {
                Debug.Log("Running with visible IDs on screen.");
                Monitor.SetActive(true);
                for (int i = 0; i < allZones.Count; i++)
                {
                    Monitor.Log("", i.ToString(), allZones[i].transform);
                }
            }
        }


        DestroyExistingVehiclesForReset();
        ClearZoneCountForReset();
        // If read_config = False, will do random spawning.
        // Else config can specify random, subset, fixed sampling.
        if (agentSpawningMethod == (int)AgentSpawning.Fixed)
        {
            FixedAgentSpawning();
        }
        else
        {
            RandomAndSubsetSpawning();
        }
    }

    public override void AcademyStep()
    {
        print("Count: ");
    }

    protected override void OnDestroy()
    {
        
    }
}

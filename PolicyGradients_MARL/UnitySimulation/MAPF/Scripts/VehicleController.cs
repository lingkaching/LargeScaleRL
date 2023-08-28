/* Vehicle Controller (Agent)
 * @Author: Tarun Gupta (tarung@smu.edu.sg)
 * Singapore Management University
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System;

public class VehicleController : Agent
{
    private GameObject currentZone;
    private GameObject destZone;
    private Vector3 destPosition;
    private int timeToDest;
    private GameObject[] goalZones;
    private int currentGoalZoneIndex;
    private GameObject currentGoalZone;
    private bool mover;
    //private bool didMove;
    private Color original;
    public float delayToDestroy;
    private int agentID;
    public static float precision = 1e-5f;

    public void SetVehicleID(int id)
    {
        this.agentID = id;
    }

    public int GetVehicleID()
    {
        return agentID;
    }

    public void SetGoalZones(GameObject[] goals)
    {
        this.goalZones = goals;
    }

    public void SetDestZoneAndTime(GameObject dest, int timeDest)
    {
        this.destZone = dest;
        this.timeToDest = timeDest;
    }

    public GameObject GetDestZone()
    {
        return this.destZone;
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Currently the goal zones have to be reached in order for the reward to be given.
    // Can change it to -- goal zones could be reached in any order.
    public override void AgentReset()
    {
        Debug.Log("Agent Reset Called");
        ResetMoverAndDest();

        if (this.GetComponent<Agent>().IsMaxStepReached())
        {
            // Should not happen since not setting the limit in prefab.
            this.GetComponent<Renderer>().material.color = Color.red;
            //Destroy(this.gameObject, delayToDestroy);
            return;
        }

        if (currentGoalZoneIndex < goalZones.Length)
        {

            this.GetComponent<Renderer>().material.color = original;
            currentGoalZone = goalZones[currentGoalZoneIndex++];
        }
        else
        {
            this.GetComponent<Renderer>().material.color = new Color(100, 100, 0);
            currentGoalZone = null;
            //Destroy(this.gameObject, delayToDestroy);
        }
    }

    public List<float> GetCurrentStateOfAgent()
    {
        List<float> state = new List<float>
        {
            agentID
        };
        if (currentZone != null)
        {
            state.Add(currentZone.GetComponent<ZoneController>().GetZoneID());
        }
        else
        {
            state.Add(-1f);
        }
        if (destZone != null)
        {
            state.Add(destZone.GetComponent<ZoneController>().GetZoneID());
            state.Add(timeToDest);
        }
        else
        {
            state.Add(-1f);
            state.Add(-1f);
        }
        if (currentGoalZone != null)
        {
            state.Add(currentGoalZone.GetComponent<ZoneController>().GetZoneID());
        }
        else
        {
            state.Add(-1f);
        }
        return state;
    }

    public override void CollectObservations()
    {
        int neighbourCount = 0;
        AddVectorObs(GetCurrentStateOfAgent());

        if (currentZone != null)
        {
            neighbourCount++;
            AddVectorObs(new List<float>
            {
                currentZone.GetComponent<ZoneController>().GetZoneID(),
                currentZone.GetComponent<ZoneController>().GetCurrentCountOfAgents()
            });
            foreach (GameObject neighbour in currentZone.GetComponent<ZoneController>().GetNeighbourZones())
            {
                neighbourCount++;
                AddVectorObs(new List<float>
                {
                    neighbour.GetComponent<ZoneController>().GetZoneID(),
                    neighbour.GetComponent<ZoneController>().GetCurrentCountOfAgents()
                });
            }
        }
        while (neighbourCount < ZoneController.maxNeighbours)
        {
            neighbourCount++;
            AddVectorObs(new List<float>
            {
                -1f,-1f
            });
        }
    }

    private int TransitionSampling(List<int> transitionInfo, List<float> samplingParams)
    {
        // Use sampling params however you like!
        return UnityEngine.Random.Range(transitionInfo[0], transitionInfo[1] + 1);
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        Debug.Log("Action Called");
        GameObject destZoneCopy = new GameObject();
        int destinationZoneIDPolicy = Mathf.FloorToInt(vectorAction[0]);
        int sampledTimeToDest = Mathf.FloorToInt(vectorAction[1]);
        List<float> samplingParameters = new List<float>();
        samplingParameters.Clear();

        for(int i=1; i<vectorAction.Length; i++)
        {
            samplingParameters.Add(vectorAction[i]);
        }

        if (currentZone == null || currentGoalZone == null)
        {
            if (Mathf.FloorToInt(destinationZoneIDPolicy) != -1)
            {
                throw new System.Exception("Invalid action -- Agent has no currentZone or currentGoalZone. Optimal suggested action: [-1,-1].");
            }
        }
        else
        {
            // Either you need an action -- next dest, time
            if (destZone == null)
            {
                bool correctness_flag = true;
                // Correct action -- prescribed a destination zone and time.
                if (destinationZoneIDPolicy != -1)
                {
                    // Check for the sampling parameters input here based on definition of action.

                    // Is destination Zone ID valid?
                    GameObject destZonePolicy = TrainAcademy.GetZoneByID(destinationZoneIDPolicy);
                    correctness_flag &= destZonePolicy != null;

                    List<int> transitionInfo = null;
                    // Is destination Zone ID a valid neighbour?
                    if (currentZone != null)
                    {
                        correctness_flag &= Array.IndexOf(currentZone.GetComponent<ZoneController>().GetNeighbourZones(), destZonePolicy) >= 0;

                        // Checking if destZonePolicy transition data is available.
                        transitionInfo = currentZone.GetComponent<ZoneController>().GetTransitionInfo(destZonePolicy);
                        correctness_flag &= transitionInfo != null;
                    }
                    else
                    {
                        correctness_flag = false;
                    }

                    // Set dest zone and time now.
                    if (correctness_flag)
                    {
                        this.destZone = destZonePolicy;
                        // this.timeToDest = TransitionSampling(transitionInfo, samplingParameters);
                        this.timeToDest = sampledTimeToDest;
                        Debug.Log("Actions Set: " + destZone + " " + timeToDest);

                        if (!mover)
                        {
                            Debug.Log("Mover set");
                            destPosition = GetDestPositionOfZone();

                            //IMPORTANT: If max possible time to dest is too high such as > 1e4, change the collider size of the vehicle accordingly.
                            var percent_of_movement = 1.0f / (float)this.timeToDest;
                            Debug.Log("Moving " + percent_of_movement);
                            this.transform.position = Vector3.Lerp(this.transform.position, destPosition, percent_of_movement);
                            Debug.Log("Moved to " + this.transform.position);
                            this.timeToDest -= 1;
                            mover = true;
                            if (this.timeToDest <= 0)
                            {
                                Debug.Log("Resetting");
                                if (destZone.CompareTag("IZone"))
                                {
                                    this.transform.position = destZone.transform.position;
                                }
                                destZoneCopy = destZone;
                                destZone.GetComponent<ZoneController>().SendMessage("RemoveVehicle", this.gameObject);
                                ResetMoverAndDest();
                            }
                        }
                    }
                    else
                    {
                        throw new System.Exception("Invalid action -- Possible Causes: \n1) Invalid destination zone\n2) Destination zone not valid neighbour of current zone\n3) Transition Info for selected z and z' not available.");
                    }
                }
                else
                {
                    // Prescribed a no-op, when action required.
                    throw new System.Exception("Invalid action -- Prescribed a no-op, when action required.");
                }
            }
            else // Or you don't need an action -- no-op
            {
                // You don't need action but still given -- error in policy, should have been a no-op.
                if (Mathf.FloorToInt(destinationZoneIDPolicy) != -1)
                {
                    throw new System.Exception("Invalid action -- Agent already has a destination. Optimal Suggested Action: [-1,-1]");
                }
                else
                {
                    if (!mover)
                    {
                        throw new System.Exception("Mover should have been already set.");
                    }
                    else
                    {
                        Debug.Log("Mover already set");

                        //IMPORTANT: If max possible time to dest is too high such as > 1e4, change the collider size of the vehicle accordingly.
                        var percent_of_movement = 1.0f / (float)this.timeToDest;
                        Debug.Log("Moving " + percent_of_movement);
                        this.transform.position = Vector3.Lerp(this.transform.position, destPosition, percent_of_movement);
                        Debug.Log("Moved to " + this.transform.position);
                        this.timeToDest -= 1;
                        if (this.timeToDest <= 0)
                        {
                            Debug.Log("Resetting");
                            if (destZone.CompareTag("IZone"))
                            {
                                this.transform.position = destZone.transform.position;
                            }
                            destZoneCopy = destZone;
                            destZone.GetComponent<ZoneController>().SendMessage("RemoveVehicle", this.gameObject);
                            ResetMoverAndDest();
                        }
                    }
                }
            }
        }
        // Rewards
        if (currentGoalZone != null)
        {
            if ( destZoneCopy == currentGoalZone && this.timeToDest <= 0)
            {
                //EventManager.TriggerEvent("goal", this.gameObject);
                AddReward(10f);
                Done(); // Will call AgentReset to change the goal zone, if there are any remaining. Else make goal zone null.
            }
            else
            {
                AddReward(-0.1f);
            }
        }
        else
        {
            AddReward(0.0f);
        }
    }

    public override void InitializeAgent()
    {
        Debug.Log("Agent Initialize Called");
        currentGoalZoneIndex = 0;
        original = this.GetComponent<Renderer>().material.color;
        ResetMoverAndDest();
    }

    public static bool AlmostEquals(float double1, float double2)
    {
        return (Mathf.Abs(double1 - double2) <= precision);
    }

    public static bool AlmostEquals(Vector3 vec1, Vector3 vec2)
    {
        return (Mathf.Abs(vec1.x - vec2.x) <= precision) &&
               (Mathf.Abs(vec1.y - vec2.y) <= precision) &&
               (Mathf.Abs(vec1.z - vec2.z) <= precision);
    }

    private Vector3 GetDestPositionOfZone()
    {
        float adjustment;
        if (destZone.CompareTag("Zone"))
        {
            adjustment = 1.0f;
        }
        else if (destZone.CompareTag("IZone"))
        {
            adjustment = 0.5f;
        }
        else
        {
            adjustment = 0.0f;
            Debug.Log("Not Possible.");
        }

        if (destZone != null)
        {
            // Move to only start of a normal Zone.
            Vector3 destPost = destZone.transform.position;
            Vector3 diff_vector = destPost - this.transform.position;
            Debug.Log(this.GetStepCount() + " -- X " + diff_vector.x.CompareTo(0.0f));
            Debug.Log(this.GetStepCount() + " -- Y " + diff_vector.y.CompareTo(0.0f));
            Debug.Log(this.GetStepCount() + " -- Z " + diff_vector.z.CompareTo(0.0f));

            if (!AlmostEquals(diff_vector.x, 0.0f) && AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.z, 0.0f))
            {
                Debug.Log(this.GetStepCount() + " -- X-Condition True");
                if (diff_vector.x.CompareTo(0.0f) < 0)
                {
                    Debug.Log(this.GetStepCount() + " -- X-Condition True +1");
                    destPost.x += adjustment;
                }
                else
                {
                    Debug.Log(this.GetStepCount() + " -- X-Condition True -1");
                    destPost.x -= adjustment;
                }
            }
            else if (!AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.x, 0.0f) && AlmostEquals(diff_vector.z, 0.0f))
            {
                Debug.Log(this.GetStepCount() + " -- Y-Condition True");
                if (diff_vector.y.CompareTo(0.0f) < 0)
                {
                    Debug.Log(this.GetStepCount() + " -- Y-Condition True +1");
                    destPost.y += adjustment;
                }
                else
                {
                    Debug.Log(this.GetStepCount() + " -- Y-Condition True -1");
                    destPost.y -= adjustment;
                }
            }
            else if (!AlmostEquals(diff_vector.z, 0.0f) && AlmostEquals(diff_vector.y, 0.0f) && AlmostEquals(diff_vector.x, 0.0f))
            {
                Debug.Log(this.GetStepCount() + " -- Z-Condition True");
                if (diff_vector.z.CompareTo(0.0f) < 0)
                {
                    Debug.Log(this.GetStepCount() + " -- Z-Condition True +1");
                    destPost.z += adjustment;
                }
                else
                {
                    Debug.Log(this.GetStepCount() + " -- Z-Condition True -1");
                    destPost.z -= adjustment;
                }
            }
            else
            {
                throw new System.Exception(this.GetStepCount() + " -- NO - Condition True");
            }
            Debug.Log(this.GetStepCount() + " -- Current Zone position: " + currentZone.transform.position);
            Debug.Log(this.GetStepCount() + " -- Actual Current position: " + this.transform.position);
            Debug.Log(this.GetStepCount() +  " -- Actual destination position: " + destZone.transform.position);
            Debug.Log(this.GetStepCount() + " -- Altered destination position: " + destPost);
            Debug.Log(this.GetStepCount() + " -- Difference vector: " + diff_vector);
            Debug.Log(this.GetStepCount() + " -- Steps to reach: " + this.timeToDest);
            return destPost;
        }
        else
        {
            throw new System.Exception("Function Incorrectly called.");
            //return new Vector3(-1, -1, -1);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void ResetMoverAndDest()
    {
        mover = false;
        destZone = null;
        timeToDest = -1;
    }

    // Made collider of vehicle a box of size (1e-5, 1e-5, 1e-5).
    // So new zone will be detected only when that collider (meaning center of the vehicle) is
    // within (1e-5, 1e-5, 1e-5) precision of the actual zone destination start.
    void OnTriggerEnter(Collider other)
    {
        // Closely correlated with size of the collider as in the collider component.
        if (other.CompareTag("Zone") || other.CompareTag("IZone"))
        {
            // Remove this agent from all lists.
            EventManager.TriggerEvent("zoneChangeListener", this.gameObject);

            // New current zone assigned.
            currentZone = other.gameObject;

            // Add this agent to the new current zone.
            other.gameObject.GetComponent<ZoneController>().SendMessage("AddVehicle", this.gameObject);
        }
    }

    // Since we have to move to the center of an IZone, we delay changing current zone till vehicle's center reaches 
    // the center of the IZone within (1e-5, 1e-5, 1e-5) precision.
    //void OnTriggerStay(Collider other)
    //{
    //    // Closely correlated with size of the collider as in the collider component.
    //    // Might not be really required, might be working fine with OnTriggerEnter but not guaranteed to work there.
    //    if (other.gameObject.CompareTag("IZone"))
    //    {
    //        if (AlmostEquals(other.GetComponent<ZoneController>().transform.position, this.transform.position))
    //        {
    //            currentZone = other.gameObject;
    //        }
    //    }
    //}

    //IEnumerator MoveToPosition(Vector3 position, float timeToMove)
    //{
    //    var currentPos = this.transform.position;
    //    var t = 0f;
    //    while (t < 1)
    //    {
    //        t += Time.deltaTime / timeToMove;
    //        this.transform.position = Vector3.Lerp(currentPos, position, t);
    //        yield return null;
    //    }
    //}
}

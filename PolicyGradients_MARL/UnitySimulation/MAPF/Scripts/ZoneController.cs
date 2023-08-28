/* Zone Controller
 * @Author: Tarun Gupta (tarung@smu.edu.sg)
 * Singapore Management University
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using UnityEngine.Events;

public class ZoneController : MonoBehaviour
{
    private int zoneCapacity;
    public GameObject[] neighbours;
    private List<List<int>> neighbourTransitions;
    public List<GameObject> currentAgents;
    //private List<GameObject> decisionPendingAgents;
    private Color original;
    private int zoneID;
    public static int maxNeighbours;

    private UnityAction<GameObject> zoneListener;

    public void SetNeighbourTransitions(List<List<int>> transitions)
    {
        this.neighbourTransitions = transitions;
    }

    public void SetCapacity(int cap)
    {
        this.zoneCapacity = cap;
    }

    public int GetCapacity()
    {
        return zoneCapacity;
    }

    public int GetCurrentCountOfAgents()
    {
        return currentAgents.Count;
    }

    public void SetZoneID(int id)
    {
        this.zoneID = id;
    }

    public int GetZoneID()
    {
        return zoneID;
    }

    public List<int> GetTransitionInfo(GameObject dest)
    {
        foreach(List<int> transition in neighbourTransitions)
        {
            if (transition[0] == dest.GetComponent<ZoneController>().GetZoneID())
            {
                return new List<int>
                {
                    transition[1], 
                    transition[2] 
                };
            }
        }
        return null;
    }

    void Awake()
    {
        zoneListener = new UnityAction<GameObject>(RemoveVehicle);
    }

    void OnEnable()
    {
        EventManager.StartListening("zoneChangeListener", zoneListener);
    }

    void OnDisable()
    {
        EventManager.StopListening("zoneChangeListener", zoneListener);
    }

    public GameObject[] GetNeighbourZones()
    {
        return neighbours;
    }

    public void SetNeighbourZones(GameObject[] neighs)
    {
        this.neighbours = neighs;
    }

    void RemoveVehicle(GameObject obj)
    {
        //if (decisionPendingAgents.Contains(obj))
        //{
        //    decisionPendingAgents.Remove(obj);
        //}
        if (currentAgents.Contains(obj))
        {
            Debug.Log("Remove Vehicle called for zone " + this.zoneID + " by agent " + obj.GetComponent<VehicleController>().GetVehicleID());
            currentAgents.Remove(obj);
        }
        //Debug.Log("Some Function was called!");
    }

    //public int numberOfNeighbours;

    // Start is called before the first frame update
    void Start()
    {
        maxNeighbours = 7;
        currentAgents = new List<GameObject>();
        neighbourTransitions = new List<List<int>>();
        //decisionPendingAgents = new List<GameObject>();
        zoneCapacity = 2;
        original = this.GetComponent<Renderer>().material.color;
    }

    // Update is called once per frame
    void Update()
    {
        //if (decisionPendingAgents.Count > 0)
        //{
        //    for (int i=0; i< decisionPendingAgents.Count; i++)
        //    {
        //        VehicleController ag = decisionPendingAgents[i].GetComponent<VehicleController>();

        //        if (ag.GetDestZone() == null && neighbours.Length != 0)
        //        {
        //            //print(this + "decisionPendingAgents Setting");
        //            var select_random_neighbour = Random.Range(0, neighbours.Length);
        //            decisionPendingAgents[i].GetComponent<VehicleController>().SetDestZoneAndTime((GameObject)neighbours[select_random_neighbour], (float)Random.Range(1, 2));
        //            //print(this + "decisionPendingAgents Removing");
        //            decisionPendingAgents.RemoveAt(i);
        //            i--;
        //        }

        //    }
        //}

        if (currentAgents.Count > zoneCapacity)
        {
            this.GetComponent<Renderer>().material.color = new Color32(255,0,40,50);
        }
        else
        {
            this.GetComponent<Renderer>().material.color = original;
        }
    }

    private void AddVehicle(GameObject obj)
    {
        if (!currentAgents.Contains(obj))
        {
            Debug.Log("Add Vehicle called for zone " + this.zoneID + " by agent " + obj.GetComponent<VehicleController>().GetVehicleID());
            currentAgents.Add(obj);
        }
    }

    //private void OnTriggerEnter(Collider other)
    //{
    //    //if (!decisionPendingAgents.Contains(other.gameObject))
    //    //{
    //    //    decisionPendingAgents.Add(other.gameObject);
    //    //    //print(this + "decisionPendingAgents Adding");
    //    //}
    //    if (!currentAgents.Contains(other.gameObject))
    //    {
    //        currentAgents.Add(other.gameObject);
    //        //print(this + "currentAgents Adding");
    //    }
    //}

    private void OnTriggerExit(Collider other)
    {
        // Should not be called since exit is now based on an event.
        if (currentAgents.Contains(other.gameObject))
        {
            currentAgents.Remove(other.gameObject);
        }
    }
}

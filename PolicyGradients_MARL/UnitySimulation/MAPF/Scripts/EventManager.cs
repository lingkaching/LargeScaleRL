/* Event Manager
 * @Author: Tarun Gupta (tarung@smu.edu.sg)
 * Singapore Management University
 */

using UnityEngine;
using UnityEngine.Events;
using System.Collections;
using System.Collections.Generic;

[System.Serializable]
public class ThisEvent : UnityEvent<GameObject>
{

}

public class EventManager : MonoBehaviour
{

    private Dictionary<string, ThisEvent> eventDictionary;

    private static EventManager eventManager;

    public static EventManager instance
    {
        get
        {
            if (!eventManager)
            {
                eventManager = FindObjectOfType(typeof(EventManager)) as EventManager;

                if (!eventManager)
                {
                    Debug.LogError("There needs to be one active EventManager script on a GameObject in your scene.");
                }
                else
                {
                    eventManager.Init();
                }
            }

            return eventManager;
        }
    }

    void Init()
    {
        if (eventDictionary == null)
        {
            eventDictionary = new Dictionary<string, ThisEvent>();
        }
    }

    public static void StartListening(string eventName, UnityAction<GameObject> listener)
    {
        if (instance.eventDictionary.TryGetValue(eventName, out ThisEvent thisEvent))
        {
            thisEvent.AddListener(listener);
        }
        else
        {
            thisEvent = new ThisEvent();
            thisEvent.AddListener(listener);
            instance.eventDictionary.Add(eventName, thisEvent);
        }
    }

    public static void StopListening(string eventName, UnityAction<GameObject> listener)
    {
        if (eventManager == null) return;
        if (instance.eventDictionary.TryGetValue(eventName, out ThisEvent thisEvent))
        {
            thisEvent.RemoveListener(listener);
        }
    }

    public static void TriggerEvent(string eventName, GameObject param)
    {
        if (instance.eventDictionary.TryGetValue(eventName, out ThisEvent thisEvent))
        {
            thisEvent.Invoke(param);
        }
    }
}
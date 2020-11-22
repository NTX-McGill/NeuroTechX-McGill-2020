﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using SocketIO;

public class socketIO : MonoBehaviour
{
	private SocketIOComponent socket;
    public InputField textBar;
    public Button[] options;

	public void Start()
	{
		GameObject go = GameObject.Find("SocketIO");
		socket = go.GetComponent<SocketIOComponent>();

		socket.On("open", TestOpen);
    socket.On("word", HandleWord);
		socket.On("options", HandleOptions);
  	socket.On("delete", HandleDeleteWord);
  	socket.On("selection", HandleSelectionMode);

		socket.On("error", TestError);
  }

	public void TestOpen(SocketIOEvent e)
	{
		Debug.Log("[SocketIO] Open received: " + e.name + " " + e.data);
	}

	public void HandleWord(SocketIOEvent e){
	    textBar.text += e.data["word"].ToString().Trim('"')+" ";
        for (int i =0; i < options.Length; i++){
            options[i].GetComponentInChildren<Text>().text = "";
        }
	}

	public void HandleOptions(SocketIOEvent e){
	    Debug.Log(e.data["words"]);
	    Debug.Log(e.data["words"].Count);
        for (int i =0; i < options.Length; i++){
            string optionText = "";
            if(i < e.data["words"].Count)
                optionText = e.data["words"][i].ToString().Trim('"');
            options[i].GetComponentInChildren<Text>().text = optionText;
        }
	}

    public void HandleDeleteWord(SocketIOEvent e){
        Debug.Log("deleting word");
        string newSentence = textBar.text;
        Debug.Log(newSentence);
        newSentence = newSentence.Trim();
        int endIndex = newSentence.LastIndexOf(" ",newSentence.Length);
        if (endIndex == -1)
            textBar.text = "";
        else
            textBar.text = newSentence.Substring(0,endIndex+1);
    }

    public void HandleSelectionMode(SocketIOEvent e){
        Debug.Log("selection mode on");
    }

	public void TestError(SocketIOEvent e)
	{
		Debug.Log("[SocketIO] Error received: " + e.name + " " + e.data);
		Debug.Log(e);
	}
    // Update is called once per frame
    void Update()
    {
    }
}

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:08:54 2023

@author: nehak
"""

# Slack integration 


import slack_sdk as slack
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from slackeventsapi import SlackEventAdapter
from chat import get_response



app = Flask(__name__)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


slack_event_adapter = SlackEventAdapter(os.environ['SLACK_SIGNING_SECRET'],
                                        '/slack/events', app)


client = slack.WebClient(token=os.environ['SLACK_BOT_TOKEN'])
BOT_ID = client.api_call("auth.test")['user_id']


GREETING_MESSAGE = "Hello {user_name}, welcome to the {channel_name} " \
                   "channel! We're excited to have you here."
                   
                   
welcomed_users = set()                   
                   
@slack_event_adapter.on('member_joined_channel')                   

def handle_member_joined_channel(event_data):
    user_id = event_data['event']['user']
    channel_id = event_data['event']['channel']

    # Only send a welcome message if the user is new
    if user_id not in welcomed_users:
        welcomed_users.add(user_id)

        user_info = client.users_info(user=user_id)
        user_name = user_info['user']['name']

        channel_info = client.conversations_info(channel=channel_id)
        channel_name = channel_info['channel']['name']

        greeting = GREETING_MESSAGE.format(user_name=user_name,
                                          channel_name=channel_name)

        client.chat_postMessage(channel=channel_id, text=greeting)
                          
        
@slack_event_adapter.on('message')
def message(payload):
    print(payload)
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    print(text)
    
    out = get_response(text)
 
    if BOT_ID != user_id:
        client.chat_postMessage(channel=channel_id,text=out)
 

if __name__ == "__main__":
    app.run(debug=True, port=5000)
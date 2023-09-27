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
                   
                   
                   
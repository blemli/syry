#!/usr/bin/env python3

import click
import requests
import urllib3
import os
from dotenv import load_dotenv


load_dotenv()
PHONE_IP=os.getenv('PHONE_IP')
PHONE_USER=os.getenv('PHONE_USER')
PHONE_PASSWORD=os.getenv('PHONE_PASSWORD')


# disable insecure-HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@click.command()
@click.argument('number')
@click.option('--verbose', is_flag=True, help='Show full HTTP response')
def call(number, verbose):
    """
    Trigger a Yealink T31P outgoing call via its HTTP action URI.
    """
    number=number.replace('+', '')
    number=number.replace(' ', '')
    url = (
        f"https://{PHONE_IP}/servlet?key={number}"
    )
    if verbose:
        click.echo(f'Calling {number} via {url}')
    # default HTTP Basic Auth credentials
    auth = (os.getenv("PHONE_USER"), os.getenv('PHONE_PASSWORD'))
    resp = requests.get(url, auth=auth, verify=False)

    if verbose:
        click.echo(f'HTTP {resp.status_code}\n\n{resp.text}')
    else:
        click.echo(resp.status_code)

if __name__ == '__main__':
    call()

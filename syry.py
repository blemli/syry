#!/usr/bin/env python3

import click
import requests
import urllib3
import os
import platform
import re
import llm
import faster_whisper as fw
from dotenv import load_dotenv
from listen import listen_macos, listen_linux

load_dotenv()
PHONE_IP=os.getenv('PHONE_IP')
PHONE_USER=os.getenv('PHONE_USER')
PHONE_PASSWORD=os.getenv('PHONE_PASSWORD')
KURT_IP=os.getenv('KURT_IP')

# disable insecure-HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@click.group()
def cli():
    """Syry - Phone and audio utility tool"""
    pass

@cli.command("call")
@click.argument('number')
@click.option('--verbose', is_flag=True, help='Show full HTTP response')
def call_cli(number, verbose):
    """
    Command-line interface for calling a number.
    This function triggers a call to the specified number using the Yealink T31P phone.
    """
    call(number, verbose)
    
    
def call(number, verbose):
    """
    Trigger a Yealink T31P outgoing call via its HTTP action URI.
    """
    #remove all non numeric characters
    number = re.sub(r'\D', '', number)
    url = f"https://{PHONE_IP}/servlet?key={number}"
    if verbose:
        click.echo(f'Calling {number} via {url}')
    # default HTTP Basic Auth credentials
    auth = (os.getenv("PHONE_USER"), os.getenv('PHONE_PASSWORD'))
    resp = requests.get(url, auth=auth, verify=False)
    if verbose:
        click.echo(f'HTTP {resp.status_code}\n\n{resp.text}')

@cli.command('listen')
@click.option('--model', default='turbo', help='Whisper model to use for transcription')
@click.option('--verbose', is_flag=True, help='Show diagnostic information')
def listen_for_contact(model,verbose):
    if platform.system() == 'Darwin':
        transcription=listen_macos(model)
    elif platform.system() == 'Linux':
        transcription=listen_linux(model)
    number=select_number(transcription)
    if verbose: click.echo(f'number: {number}')
    if number:
        click.echo(f'Calling {number}...')
        print(type(number))
        call(number,verbose=verbose)
    else:
        click.echo('No number found.')
    
def get_addressbook():
    """
    Get the address book from Kurt.
    """
    url=f"http://{KURT_IP}:8000/phonebook.xml"
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            return response.text
        else:
            click.echo(f"Error fetching address book: {response.status_code}")
            return None
    except requests.RequestException as e:
        click.echo(f"Error fetching address book: {e}")
        return None

def select_number(transcription):
    """
    Use Gemma 2 2B model to analyze the transcription and select a phone number from the address book.
    
    Args:
        transcription (str): The transcribed text from the user's speech
        
    Returns:
        str: Selected phone number or None if no match is found
    """
    return _select_number_impl(transcription)

@cli.command('select')
@click.argument('transcription')
def select_number_cmd(transcription):
    """
    Command-line interface for select_number function.
    Select a phone number from the address book based on transcription.
    """
    result = _select_number_impl(transcription)
    if result:
        click.echo(result)
    return result

def _select_number_impl(transcription):
    """
    Implementation of the number selection logic.
    
    Args:
        transcription (str): The transcribed text from the user's speech
        
    Returns:
        str: Selected phone number or None if no match is found
    """
    addressbook = get_addressbook()
    if not transcription or not addressbook:
        click.echo("Missing transcription or address book data.")
        return None
    
    try:
        # Import required packages
        from lxml import etree
        
        click.echo("Parsing address book...")
        # Parse the XML address book
        try:
            root = etree.fromstring(addressbook.encode('utf-8'))
            entries = []
            
            # Extract entries from XML
            for entry in root.findall(".//DirectoryEntry"):
                name_elem = entry.find("Name")
                tel_elems = entry.findall("Telephone")
                
                if name_elem is not None and name_elem.text and tel_elems:
                    name = name_elem.text.strip()
                    phones = [tel.text.strip() for tel in tel_elems if tel.text]
                    if name and phones:
                        entries.append({"name": name, "phones": phones})
        except Exception as e:
            click.echo(f"Error parsing address book XML: {e}")
        
        if not entries:
            click.echo("No valid entries found in the address book.")
            return None
            
        click.echo(f"Found {len(entries)} contacts in the address book.")
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are an AI assistant helping to find a contact in a phone directory based on a spoken request.

The user said: "{transcription}"

Available contacts:
"""
        for i, entry in enumerate(entries):
            phones_str = ", ".join(entry["phones"])
            prompt += f"{i+1}. {entry['name']} - {phones_str}\n"
            
        prompt += """
Based on the user's request, which contact should be called? If multiple phone numbers are available for a contact, select the most appropriate one (mobile preferred for personal contacts, office/main number for businesses).

Return ONLY the phone number to call, including country code. If no match is found or the request is unclear, return "NONE".
"""
        
        click.echo("Querying Gemma 2 2B model...")
        
            # Call the Gemma 2 2B model using the llm package
        try:
            # Using the correct llm API to access the Gemma model
            response = llm.get_model("gemma3:12b").prompt(prompt)
            result = str(response).strip()
            click.echo(f"Model response: {result}")
            
            # Process the response to extract a valid phone number
            import re
            phone_match = re.search(r'[\+]?[0-9]{10,}', result)
            
            if phone_match:
                selected_number = phone_match.group(0)
                click.echo(f"Selected number: {selected_number}")
                return selected_number
            elif "NONE" in result.upper():
                click.echo("No matching contact found by the model.")
                return None
            else:
                # Try to match the response with phone numbers in entries
                for entry in entries:
                    for phone in entry["phones"]:
                        if phone in result:
                            click.echo(f"Found matching number in response: {phone}")
                            return phone
                
                click.echo("Could not extract a valid phone number from the model response.")
                return None
                
        except Exception as e:
            click.echo(f"Error querying the Gemma model: {e}")
            
            # Fallback: Simple keyword matching as a backup method
            transcription_lower = transcription.lower()
            best_match = None
            best_score = 0
            
            for entry in entries:
                name_lower = entry["name"].lower()
                # Simple keyword matching score (number of words from name found in transcription)
                name_words = name_lower.split()
                score = sum(1 for word in name_words if word in transcription_lower)
                
                if score > best_score and entry["phones"]:
                    best_score = score
                    # Prefer first number (usually main number)
                    best_match = entry["phones"][0]
            
            if best_match:
                click.echo(f"Fallback method found number: {best_match}")
                return best_match
            
            return None
            
    except Exception as e:
        click.echo(f"Error in select_number: {e}")
        return None
    
    


if __name__ == '__main__':
    cli()

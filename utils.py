import requests 
import os
def get_file(local_file, remote_file):
    """ downloads remote_file to local_file if necessary """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)  
        
        
    
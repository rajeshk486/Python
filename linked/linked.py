# insert your application KEY and SECRET

API_KEY = "819b9ld1jymobk"
SECRET_KEY = "sBPyr2wY3OWWI8i3"

import webbrowser

from linkedin.linkedin import LinkedInAuthentication

li = LinkedInAuthentication(API_KEY, SECRET_KEY)

token = li.getRequestToken(None)

# prompt user in the web browser to login to LinkedIn and then enter a code that LinkedIn gives to the user

auth_url = li.getAuthorizeUrl(token)
webbrowser.open(auth_url)
validator = input("Enter token: ")
access_token=li.getAccessToken(token,validator)

# list all connections

connections = li.connections_api.getMyConnections(access_token)

print("number of connections:{} ".format( len(connections)))
for c in connections:
    print("{}  {}".format(c.firstname, c.lastname))
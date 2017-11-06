from PyLinkedinAPI.PyLinkedinAPI import PyLinkedinAPI
from linkedin.linkedin import LinkedInAuthentication,LinkedInApplication,PERMISSIONS

port = 8000
API_KEY = "wFNJekVpDCJtRPFX812pQsJee-gt0zO4X5XmG6wcfSOSlLocxodAXNMbl0_hw3Vl"
API_SECRET = "daJDa6_8UcnGMw1yuq9TjoO_PMKukXMo8vEMo7Qv5J-G3SPgrAV0FqFCd0TNjQyG"
RETURN_URL = "http://localhost:8000"
auth = LinkedInAuthentication(API_KEY, API_SECRET, 'http://localhost:8000/',
                              PERMISSIONS.enums.values())
app = LinkedInApplication(authentication=auth)
print auth.authorization_url_wait_for_user_to_enter_browser(app, port)

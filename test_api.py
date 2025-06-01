import requests
import json

# Correct URL with the function name
url = "https://api.cortex.cerebrium.ai/v4/p-714e82e6/mtailor1/predict"

# Your deployment expects image_url, not prompt
payload = json.dumps({
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg"
})

headers = {
    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTcxNGU4MmU2IiwiaWF0IjoxNzQ4NzM4NTA1LCJleHAiOjIwNjQzMTQ1MDV9.k3-g9FloD0xXAvLqHILm7hi6ApySEt_gvSyuzNWo9NVm6iFjE6hanjWCPIGw6V21c6_ci6r7o-zsfPggLzaxSreFhYOpEzlbkcZVnJQ6GUF5cD_R_ZUIkUwdtqR3kX2aLSg327OuABpDAE2QVund-n-AuwRxmv9HeLDqdIHdRAcfmtYcTLv4tEAPJgaWtLB1V-PDnJUu-zWZCDWhmncumIjwpYAsLgtuvaFNDcPjg1LXYoCJSKCn0VoV1z9GqMwR7V9Dyj8JI3JawjwEwsrk4ihEs70pN-BSlyFHLReXB_bzLwrWhWbQ9DvDgUrhN5imZf6-rrhXlwkgyFto_5nhLQ',
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
from google import genai

# Make sure to configure your API key
client = genai.Client(api_key="")

for model in client.models.list():
  print(model.name)
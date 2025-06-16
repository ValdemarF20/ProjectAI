from google import genai

client = genai.Client(api_key="AIzaSyBxQbP2ukABclpD07OYAmHzBcNSIBfuGzc")
print(client.models.list())
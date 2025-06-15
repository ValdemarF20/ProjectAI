import time

from google import genai
from pathlib import Path

# Model: Gemma 3 (Google)
# Parameters: 4 billion
# Type: Instruction Tuned

client = genai.Client(api_key="AIzaSyBxQbP2ukABclpD07OYAmHzBcNSIBfuGzc")
model_4 = "gemma-3-4b-it"
model_12 = "gemma-3-12b-it"
prompt = "What 5 characters are shown in this CAPTCHA?  \
          Answer ONLY the string (5 characters)."
delay = 2.5 # seconds between requests to avoid rate limiting

def get_model_type(model: str):
    if model == model_4:
        return "4"
    elif model == model_12:
        return "12"
    else:
        return "-1"

def get_img_accuracy(response_text: str, img_text: str) -> float:
    # Method must check for correct chars in exact positions
    # and return the ratio of correct characters to total characters
    if not response_text or not img_text:
        return 0.0

    response_text = response_text.strip()
    correct_chars = 0
    for i in range(5):
        if response_text[i] == img_text[i]:
            correct_chars += 1
    return correct_chars / 5

def check_model(model: str, image: str):
    file = client.files.upload(file=f"data/samples/{image}.png")

    response = client.models.generate_content(
        model=model, contents=[prompt, file]
    )
    response_text = response.text

    # Retry until we get a response with exactly 5 characters
    retries = 0
    errors = 0
    while len(response_text) != 5:
        time.sleep(delay) # Sleep to avoid rate limiting
        response = client.models.generate_content(
            model=model, contents=[prompt, file]
        )
        response_text = response.text
        retries += 1
        if retries > 5:
            print("Failed to get a valid response after 5 retries. For image:", image)
            with open(f"results_{get_model_type(model)}.txt", "a") as f:
                try:
                    f.write(f"{image};{response_text}-INVALID;0.0\n")
                except Exception as e:
                    print(f"Error writing to results_{get_model_type(model)}.txt: {e} - response: {response_text}")
                    errors += 1
                    if errors < 5:
                        continue # Try image again
            return image, "Invalid response", 0.0

    # Get the accuracy of the response (ratio of correct characters)
    accuracy = get_img_accuracy(response.text, image)

    # Save result to results.txt
    while True:
        with open(f"results_{get_model_type(model)}.txt", "a") as f:
            try:
                f.write(f"{image};{response.text};{accuracy}\n")
                break
            except Exception as e:
                print(f"Error writing to results_{get_model_type(model)}.txt: {e} - response: {response_text}")
                return image, "Invalid response", 0.0

    print(response.create_time)

    return image, response.text, accuracy

def run_all(model: str, files: list = Path("data/samples").iterdir()):
    # Get images from data directory
    counter = 1
    for file in files:
        if file.is_file() and file.suffix == ".png":
            img_text = file.stem
            print(f"Check {counter}: {check_model(model, img_text)}")
            counter += 1
            time.sleep(delay)  # Sleep to avoid rate limiting
    print("All checks completed.")

def run_from_prev(model: str):
    # Read results from results.txt and print them
    missing = Path("data/samples").iterdir()
    if not Path(f"results_{get_model_type(model)}.txt").exists():
        print("No previous results found.")
    else: # Don't run already checked images
        with open(f"results_{get_model_type(model)}.txt", "r") as f:
            lines = f.readlines()
            prev_amount = len(lines)
            print(f"Previous results found: {prev_amount} entries.")
            # Remove already checked images from the list
            for line in lines:
                img_name = line.split(";")[0]
                missing = [f for f in missing if f.stem != img_name]
    run_all(model, missing)

run_from_prev(model_12) # Continues from previous results
#run_all() # Ignores previous results
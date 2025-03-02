import openai
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_openai_client(api_key):
    client = openai.OpenAI(api_key=api_key)
    return client

def detect_object_openai(client, image_data):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages = [
                {
                    "role": "system",
                    "content": "You are a computer program that can identify objects in images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please identify the objects in the image below. Ignore humans. "
                                    "Respond **only** with a JSON list, nothing else. "
                                    "Each object should have the following fields: 'brand', 'name', 'material', and 'confidence'. "
                                    "Use 'Not Identified' for unknown fields.\n\n"
                                    "Example valid response:\n"
                                    "[\n"
                                    "    {\"brand\": \"apple\", \"name\": \"iphone\", \"material\": \"glass\", \"confidence\": 0.9},\n"
                                    "    {\"brand\": \"samsung\", \"name\": \"galaxy\", \"material\": null, \"confidence\": 0.8}\n"
                                    "]"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        }
                    ]
                }
            ]

        )
        print(response.choices[0].message.content)
        logger.info(f"Prompt given successfully", response.choices[0].message.content)
        content = response.choices[0].message.content
        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"Error when prompt was given: {e}")
        return None
    
    
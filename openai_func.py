import openai

def setup_openai_client(api_key):
    client = openai.OpenAI(api_key=api_key)
    return client

def detect_object_openai(client, image_data):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You need to detect the object in the image and provide me just a jsonl output which would have the following fields. 'detections' which is a list of objects detected. Each object should have 'id', 'product_name', 'product_company', 'material', 'confidence', fields."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error when prompt was given: {e}")
        return None
    
    
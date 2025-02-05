import json
def filter_detections(input_json):
    """
    Filters the input JSON data to extract unique objects and their highest confidence scores.

    Args:
        input_json (list): The input JSON data containing detections.

    Returns:
        str: A JSON string with the format [{"object": class_name, "confidence": confidence}].
    """
    # Dictionary to store the highest confidence for each object
    unique_detections = {}

    # Iterate through each frame in the input JSON
    for frame in input_json:
        for detection in frame.get("detections", []):
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # Update the confidence if the object is already in the dictionary
            if class_name not in unique_detections or confidence > unique_detections[class_name]:
                unique_detections[class_name] = confidence

    # Convert the dictionary to the desired output format
    filtered_detections = [
        {"object": class_name, "confidence": confidence}
        for class_name, confidence in unique_detections.items()
    ]

    # Convert the list to a JSON string
    return json.dumps(filtered_detections, indent=4) 

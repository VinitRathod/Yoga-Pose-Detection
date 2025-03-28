import json 
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
import json
import os
_data_columns = ""
_keypoints = ""
_model = ""
# from google.colab.patches import cv2_imshow
# from google.colab import drive, files
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

################ Define Global variables Start ################
# Load the MoveNet model from TensorFlow Hub
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Define the mapping of keypoints to body parts
keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

keypoint_names_dir = {
        0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
        5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
        9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
        13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"
    }

# Define the connections between keypoints to draw lines for visualization
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
               (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

################ Define Global variables End ################

################ Convert Images to Keypoints Start ################
def detect_pose_static(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192)
    image_np = image_resized.numpy().astype(np.int32)
    outputs = movenet.signatures["serving_default"](tf.constant(image_np))
    keypoints = outputs['output_0'].numpy()
    return keypoints
################ Convert Images to Keypoints End ################

################ Calculate Image to Keypoints Start ################
def calculate_angle(a, b, c):
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # End joint

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

# Function to check if an angle is within an acceptable range
def is_within_tolerance(expected, actual, tolerance):
    return abs(expected - actual) <= tolerance

# Tolerance Levels
tolerance_levels = {
    "Beginner": 50,       # More forgiving (±20°)
    "Intermediate": 15,   # Moderate strictness (±15°)
    "Advanced": 10        # Strictest (±10°)
}

# Custom tolerance per joint (default for Advanced)
default_tolerance_ranges = {
    "Left Elbow": 10, "Right Elbow": 10,
    "Left Knee": 15, "Right Knee": 15,
    "Left Shoulder": 12, "Right Shoulder": 12,
    "Left Hip": 18, "Right Hip": 18
}

# Function to update tolerance based on difficulty level
def adjust_tolerance_levels(level="Advanced"):
    base_tolerance = tolerance_levels.get(level, 10)  # Default to "Advanced" if level is invalid
    scale_factor = base_tolerance / 10  # Scaling based on Advanced (default)

    return {joint: int(value * scale_factor) for joint, value in default_tolerance_ranges.items()}


def provide_correction_feedback(detected_keypoints, reference_keypoints, predicted_pose_name, skill_level="Intermediate"):
    feedback = []

    # Critical joints (shoulder, elbow, knee, ankle)
    critical_joints = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]


    # Get the overall tolerance level based on skill level
    default_tolerance = adjust_tolerance_levels(skill_level) # Default to "Intermediate"

    # Extract reference keypoints for the predicted pose
    handstand_reference = next(
        (ref["keypoints"][0][0] for ref in reference_keypoints if ref["pose"] == predicted_pose_name),
        None
    )

    if handstand_reference is None:
        return ["Error: No reference keypoints found for the predicted pose."]

    # Ensure valid keypoint structure
    if len(detected_keypoints) < 17 or len(handstand_reference) < 17:
        return ["Error: Invalid keypoint data."]

    for joints in critical_joints:
        try:
            detected_angle = calculate_angle(
                detected_keypoints[joints[0]], detected_keypoints[joints[1]], detected_keypoints[joints[2]]
            )
            reference_angle = calculate_angle(
                handstand_reference[joints[0]], handstand_reference[joints[1]], handstand_reference[joints[2]]
            )

            # Get custom tolerance for this joint or use default
            joint_name = keypoint_names_dir[joints[1]]
            tolerance = adjust_tolerance_levels(skill_level).get(joint_name, default_tolerance[joint_name])

            # Check if the angle is within the allowed range
            if not is_within_tolerance(reference_angle, detected_angle, tolerance):
                feedback.append(
                    f"Adjust angle at {joint_name}: Expected {reference_angle:.2f}°, got {detected_angle:.2f}° (Tolerance: ±{tolerance}°)."
                )

        except (IndexError, KeyError, TypeError) as e:
            return [f"Error: Missing or incorrect keypoints for joints {joints}. Exception: {str(e)}"]

    return feedback if feedback else ["Pose is correct!"]

# Function to determine color based on deviation
def get_color_based_on_tolerance(expected, actual, tolerance):
    deviation = abs(expected - actual)
    if deviation <= tolerance:
        return (0, 255, 0)  # Green (Correct)
    elif deviation <= tolerance * 1.5:
        return (0, 255, 255)  # Yellow (Warning)
    else:
        return (0, 0, 255)  # Red (Needs correction)
################ Calculate Image to Keypoints End ################

################ Draw Keypoint in Upload Image Start ################
def visualize_pose_static(image_path,imageName, keypoints, reference_keypoints=None, predicted_pose_name="", skill_level="Intermediate"):
    image = cv2.imread(image_path)
    keypoints = np.array(keypoints)

    # Debugging: Print shape
    #print("Original keypoints shape:", keypoints.shape)

    # Adjust shape dynamically
    if len(keypoints.shape) == 4:
        keypoints = keypoints[0, 0]  # Extract if shape is (1,1,17,3)
    elif len(keypoints.shape) == 3:
        keypoints = keypoints[0]  # Extract if shape is (1,17,3)

    # Debugging: Check new shape
    #print("Processed keypoints shape:", keypoints.shape)

    # Check for NaN values
    keypoints = np.nan_to_num(keypoints)

    height, width, _ = image.shape

    # Default color (if no reference is given)
    default_color = (0, 255, 0)  # Green

    # Get corrections if reference keypoints exist
    feedback = []
    tolerance = 10  # Default angle tolerance
    default_tolerance = adjust_tolerance_levels(skill_level)  # Adjust per skill level

    if reference_keypoints is not None:
        feedback = provide_correction_feedback(keypoints, reference_keypoints, predicted_pose_name, skill_level)

    # Draw keypoints
    for i, kp in enumerate(keypoints):
        x = int(kp[1] * width)
        y = int(kp[0] * height)

        # Determine color based on feedback
        color = default_color  # Default to green

        if reference_keypoints is not None:
            for msg in feedback:
                if keypoint_names_dir[i] in msg:
                    expected_angle = float(msg.split("Expected")[1].split("°")[0].strip())
                    actual_angle = float(msg.split("got")[1].split("°")[0].strip())
                    tolerance = default_tolerance.get(keypoint_names_dir[i], 10)
                    color = get_color_based_on_tolerance(expected_angle, actual_angle, tolerance)

        cv2.circle(image, (x, y), 12, color, -1)

    # Draw connections with colors
    connections = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (5, 6), (11, 12), (5, 11), (6, 12)  # Torso
    ]

    for connection in connections:
        idx1, idx2 = connection
        x1, y1 = int(keypoints[idx1, 1] * width), int(keypoints[idx1, 0] * height)
        x2, y2 = int(keypoints[idx2, 1] * width), int(keypoints[idx2, 0] * height)

        # Determine line color based on deviation
        line_color = default_color  # Default to green

        if reference_keypoints is not None:
            for msg in feedback:
                if keypoint_names_dir[idx1] in msg or keypoint_names_dir[idx2] in msg:
                    expected_angle = float(msg.split("Expected")[1].split("°")[0].strip())
                    actual_angle = float(msg.split("got")[1].split("°")[0].strip())
                    tolerance = default_tolerance.get(keypoint_names_dir[idx1], 10)
                    line_color = get_color_based_on_tolerance(expected_angle, actual_angle, tolerance)

        cv2.line(image, (x1, y1), (x2, y2), line_color, 4)

    # cv2_imshow(image)  # Display in Colab
    filepath = os.path.join('keypointsImages', imageName)
    cv2.imwrite(filepath, image)  # Save image

    # Print corrections if needed
    if feedback:
        print("\nCorrections Needed:")
        return {"feedback": feedback,'imagePath':image_path}
        # for msg in feedback:
        #     print(msg)


################ Draw Keypoint in Upload Image End ################

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    with open("artifacts/PostFeedback.json", "r") as f:
        columns = json.load(f)
    with open("artifacts/yoga_keypoints.json", "r") as f:
        keyPoints = json.load(f)    
    global _data_columns
    global _keypoints
    global _model
    _data_columns = columns
    _keypoints = keyPoints
    # print(__keypoints)
    # global __model
    with open("artifacts/yoga_pose_detection.pickle", "rb") as f:
        _model = pickle.load(f)
    print("Loading saved artifacts...done")

def get_pose_feedback(pose_name):
    return _data_columns.get(pose_name, "Pose not found.")

################ Load Saved Artifacts Start ################
def predict_pose(image_path):
    keypoints = detect_pose_static(image_path)  # Your pose detection function
    keypoints_flat = keypoints.flatten().reshape(1, -1)
    prediction = _model.predict(keypoints_flat)  # Your trained classifier
    return prediction[0], keypoints
################ Load Saved Artifacts End ################

def posePrediction(imagePath,imageName,level):
    # if not os.path.exists(imageName):
    #     print(f"Error: The uploaded image '{imageName}' does not exist.")
    #     exit()

    reference_keypoints=_keypoints
    predicted_pose, keypoints = predict_pose(imagePath)
    print(f"\nModel Predict as the pose is : {predicted_pose}")
    correctionsFeedback = visualize_pose_static(imagePath, imageName, keypoints, reference_keypoints,predicted_pose,level)
    feedback = get_pose_feedback(predicted_pose)

    print("###########################################")

    print("Advantages:", "\n".join(feedback["advantages"]))
    print("Injury Risks:", "\n".join(feedback["risks"]))
    return {'feedback':feedback,'correctionsFeedback':correctionsFeedback,'predictedPose':predicted_pose}

if __name__ == '__main__':
    load_saved_artifacts()
    # print(__data_columns)
    # print(__model)
    # print(get_feedbacks())
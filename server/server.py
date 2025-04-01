from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import util

app = Flask(__name__,template_folder=os.path.abspath("../client"))
app.config['UPLOAD_PATH'] = 'images'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']

@app.route('/')
def hello():
    return render_template('index.html')

# def read_json():
#     response = jsonify({
#         'feedbacks':util.get_pose_feedback("Cat-Cow Stretch")
#     })

#     response.headers.add('Access-Control-Allow-Origin', '*')
#     return response
#     # with open('data.json') as f:
#     #     data = f.read()
#     # return data

# @app.route('/get_feedback', methods=['GET'])
# def get_feedback():
#     return read_json()

@app.route('/get_pose', methods=['POST'])
def get_pose():
    imagePath = ""
    image = request.files['image']
    level = request.form['level']
    imageName = secure_filename(image.filename)
    # return image.
    # image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
    if imageName != '':
        file_ext = os.path.splitext(imageName)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return "Invalid Image", 400
        imagePath = os.path.join(app.config['UPLOAD_PATH'], imageName)
        image.save(imagePath)
        pose = util.posePrediction(imagePath, imageName, level)

        return pose, 200
    else:
        return "Image not found", 400
    
    # print(image)
    
    # print(level)
    # return "Testing"
    # return image,level
    # image = data['images']
    # level = data['level']
    # return jsonify({
    #     'feedback': util.get_pose_feedback(pose)
    # })

@app.route('/keypointsImages/<path:filename>')
def serve_image(filename):
    return send_from_directory('keypointsImages', filename)

if __name__ == '__main__':
    print("Server is Running")
    # print(os.path.join(app.config['UPLOAD_PATH']))
    util.load_saved_artifacts()
    app.run(debug=True)
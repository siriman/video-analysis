from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template
from players_video import get_video
import os
import cv2
import time
from werkzeug.utils import secure_filename
# Initialize Flask application
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FILENAME'] = "test.mp4" 

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # create a secure filename
      filename = f.filename
      # filename = "test.mp4"
      app.config['VIDEO_FILENAME'] = filename
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)
      #get_video(filename)
      return render_template("uploaded.html", display_detection = filename, fname = filename)


#video_name = "./static/uploads/test.mp4"

#video_name = "./static/uploads/" + filename

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(get_video(app.config['VIDEO_FILENAME']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)

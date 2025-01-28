from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera, music_rec

app = Flask(__name__)

# Define the table headings for song recommendations
headings = ("Id", "Name", "Album", "Artist", "Cover")
video_camera = VideoCamera()  # Initialize the VideoCamera instance

@app.route('/')
def index():
    # Render the main HTML page and load default headings and empty recommendations
    recommendations = video_camera.df1  # Get the initial state of recommendations
    print(recommendations.to_json(orient='records'))  # Debugging log for recommendations
    return render_template('index.html', headings=headings, data=recommendations)

def gen(camera):
    """Generate video feed frames and update recommendations."""
    while True:
        # Continuously retrieve the latest video frame and recommendations
        frame, recommendations = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed to the client."""
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    """
    Serve the latest song recommendations in JSON format
    for the frontend to process and display dynamically.
    """
    recommendations = video_camera.df1  # Get the current recommendations
    if recommendations is None or recommendations.empty:
        print("No data available, returning default placeholder.")
        # Return placeholder data if recommendations are unavailable or invalid
        return jsonify([{
            'EmotionId': 'N/A',
            'Id': 'N/A',
            'Name': 'No Data',
            'Album': 'N/A',
            'Artist': 'N/A',
            'Cover': 'https://via.placeholder.com/150'
        }])
    else:
        print("Current Recommendations:", recommendations.to_json(orient='records'))
        return recommendations.to_json(orient='records')  # Convert DataFrame to JSON

if __name__ == '__main__':
    app.debug = True
    app.run()

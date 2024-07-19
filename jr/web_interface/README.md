## Vocalization interpretter web interface

The web interface has two components:
- `flask_audio_model.py`: An API serving an audio model using flask
- `index.html`: A webpage using javascript to record audio from the user's microphone, send it to the local API, and render the results

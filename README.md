# T5 Podcast Summariser REST API

This project is a quick and simple REST API to generate summaries using a T5 model fine-tuned on the Spotify Podcasts Dataset..

## Usage

If you want to clone this repo and try it out locally then first run:

    pip install --user -r requirements.txt

I did not build the project in a virtual environment, so there are some unnecessary dependencies in `requirements.txt` file.

`cd` into the project and run `python app.py`

If all goes well, you should be able to access the app here `http://127.0.0.1:5000/`

From a python client script, you can now generate a summary like so:

    import requests
    
    res = requests.post('http://127.0.0.1:5000/predict', json={"transcript":"This is a test podcast. What results are we going to get?"})
    
    if res.ok:
	    summary = res.json()
	    print(summary['summary'])

Running inference with this model may be slow.
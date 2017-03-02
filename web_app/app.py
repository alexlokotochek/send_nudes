import os
import requests
import urllib
import uuid
import subprocess
import cv2
import traceback
from flask import Flask, jsonify, redirect, request
import face_to_vec.porn_creator as porn_creator

WEB_APP_DIR = os.path.dirname(__file__)
app = Flask(__name__)

@app.route('/')
def index():
    return redirect("/static/index.html", code=302)

@app.route('/api', methods=['POST'])
def names():
    pk = uuid.uuid4()
    uri = 'static/images/{}.jpg'.format(pk)
    if 'X-File-Name' in request.headers:
        s = request.data
        with open(uri, 'wb') as f:
            f.write(s)
    else:
        s = request.form['url']
        subprocess.call('./wget.sh {} {}'.format(s, uri), shell=True)
    image = cv2.imread(uri, cv2.IMREAD_COLOR)
    if image is None:
        new_uri = 'static/couldnt_read.html'
    else:
        new_pk = uuid.uuid4()
        try:
            res = porn_creator.process_image(image)
        except Exception as e:
            new_uri = 'static/errors/{}.html'.format(new_pk)
            with open(new_uri, 'w') as f:
                f.write('{}{}{}'.format(e, '<br>' * 7, traceback.format_exc().replace('\n', '<br>')))
        else:
            new_uri = 'static/images/{}.jpg'.format(new_pk)
            cv2.imwrite(new_uri, res)
    if 'X-File-Name' in request.headers:
        return jsonify({"url": "/{}".format(new_uri)})
    else:
        return redirect("/{}".format(new_uri), code=302)


if __name__ == '__main__':
    app.run()


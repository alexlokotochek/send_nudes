import os
import requests
import urllib
import uuid
import subprocess
from flask import Flask, jsonify, redirect, request

app = Flask(__name__)


@app.route('/')
def index():
    return redirect("/static/index.html", code=302)

@app.route('/api', methods=['POST'])
def names():
    pk = uuid.uuid4()
    uri = 'static/images/{}.jpg'.format(pk)
    print os.path.abspath(uri)
    if 'X-File-Name' in request.headers:
        s = request.data
        print len(s)
        with open(uri, 'wb') as f:
            f.write(s)
        return jsonify({"url": "/{}".format(uri)})
    else:
        s = request.form['url']
        print s
        subprocess.call('./wget.sh {} {}'.format(s, uri), shell=True)
        return redirect("/{}".format(uri), code=302)


if __name__ == '__main__':
    app.run()


import base64
import io
import urllib.parse

from flask import jsonify, send_file
from scipy.misc import toimage


def make_response(status, result, message, payload=None):
    return jsonify({
        'result': result,
        'message': message,
        'payload': payload
    }), status


def make_404():
    return make_response(404, 'not_found', 'Requested resource was not found.')


def array_to_png_response(array, rescale=True):
    cmin = array.min() if rescale else 0.0
    cmax = array.max() if rescale else 1.0
    f = io.BytesIO()
    toimage(array, cmin=cmin, cmax=cmax).save(f, fmt='png')
    f.seek(0)

    return send_file(f, mimetype='image/png')


def image_to_png_response(image):
    f = io.BytesIO()
    toimage(image).save(f, fmt='png')
    f.seek(0)
    return send_file(f, mimetype='image/png')


def image_to_base64(image):
    f = io.BytesIO()
    toimage(image).save(f, fmt='png')
    f.seek(0)
    base64_image = base64.b64encode(f.getvalue())
    return 'data:image/png;base64,{}'.format(
        urllib.parse.quote(base64_image))


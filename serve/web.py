import click
import urllib
import tensorflow as tf
import numpy as np
import os

from flask import Flask, jsonify, request, render_template
from threading import Thread
from PIL import Image
from six.moves import _thread

from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork
from luminoth.vis import build_colormap, vis_objects

import cv2

app = Flask(__name__)


def get_image_url():
    image = request.args.get("image", None)
    if not image:
        raise ValueError
    print("image", image)
    urllib.request.urlretrieve(image, SAVE_PATH_TMP)

    with tf.gfile.Open(SAVE_PATH_TMP, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
        except (tf.errors.OutOfRangeError, OSError) as e:
            raise ValueError

    return image


def get_image():
    image = request.files.get('image')
    if not image:
        raise ValueError

    image = Image.open(image.stream).convert('RGB')
    image.save(SAVE_PATH_TMP)
    return image


def mod_image(image_path, minsize, maxsize, autofill=False):
    print(image_path)
    edit_image = cv2.imread(image_path)
    minx = 0
    miny = 0
    # make square
    if (autofill):
        height, width, _ = edit_image.shape
        maxpixels = width if width >= height else height
        square = np.zeros((maxpixels, maxpixels, 3), np.uint8)
        miny = int((maxpixels-height)/2)
        maxy = int(maxpixels-(maxpixels-height)/2)
        minx = int((maxpixels-width)/2)
        maxx = int(maxpixels-(maxpixels-width)/2)
        square[miny:maxy, minx:maxx] = edit_image
        edit_image = square

    # scale to min
    scale1 = 1
    if (minsize):
        height, width, _ = edit_image.shape
        minpixels = height if width >= height else width
        if minpixels < minsize:
            scale1 = minsize/minpixels
            edit_image = cv2.resize(edit_image, (0, 0),
                                    fx=(scale1), fy=(scale1))

    # scale to max
    scale2 = 1
    if (maxsize):
        height, width, _ = edit_image.shape
        maxpixels = width if width >= height else height
        if maxpixels > maxsize:
            scale2 = maxsize/maxpixels
            edit_image = cv2.resize(edit_image, (0, 0),
                                    fx=(scale2), fy=(scale2))
    cv2.imwrite(image_path, edit_image)
    with tf.gfile.Open(SAVE_PATH_TMP, 'rb') as f:
        try:
            image = Image.open(f).convert('RGB')
            return [image, minx, miny, scale1*scale2]
        except (tf.errors.OutOfRangeError, OSError) as e:
            raise ValueError


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/<model_name>/predict/', methods=['GET', 'POST'])
def predict(model_name):
    try:
        # TODO ADD more models
        if request.method == 'GET':
            print(request.args)
            # return jsonify(error='Use POST method to send image.', count=-1)
            total_predictions = request.args.get('total', None)
            min_prob = request.args.get('min_prob', None)
            only_number = request.args.get('only_number', False)
            id_task = request.args.get('id', False)
            force_square = request.args.get('force_square', True)

            try:
                image_array = get_image_url()
            except ValueError:
                return jsonify(error='Missing image', count=-2)
            except OSError:
                return jsonify(error='Incompatible file type', count=-3)
        else:  # POST
            total_predictions = request.form.get('total', None)
            min_prob = request.form.get('min_prob', None)
            only_number = request.form.get('only_number', False)
            id_task = request.form.get('id', False)
            force_square = request.form.get('force_square', True)
            try:
                image_array = get_image()
            except ValueError:
                return jsonify(error='Missing image', count=-2)
            except OSError:
                return jsonify(error='Incompatible file type', count=-3)
        
        if force_square == 'false':
            force_square = False
            
        if total_predictions is not None:
            try:
                total_predictions = int(total_predictions)
            except ValueError:
                total_predictions = None
        if min_prob is not None:
            try:
                min_prob = float(min_prob)
            except ValueError:
                min_prob = None
        if only_number == "False":
            only_number = None

        if not id_task:
            return jsonify(error='Missing task_id', count=-4)

        outsetx = 0
        outsety = 0
        scale = 0

        if force_square:
            image_array, outsetx, outsety, scale = mod_image(
                SAVE_PATH_TMP, None, None, True)

        # Wait for the model to finish loading.
        NETWORK_START_THREAD.join()

        objects = PREDICTOR_NETWORK.predict_image(image_array)

        if min_prob:
            objects = [obj for obj in objects if obj['prob'] >= min_prob]

        if total_predictions:
            objects = objects[:total_predictions]
        
        # Save predicted image.
        if SAVE_PATH_GLOBAL:
            path_img_vis = "{}{}_Counted.jpg".format(SAVE_PATH_GLOBAL, id_task)
            print(path_img_vis, "Saved")
            vis_objects(np.array(image_array), objects).save(path_img_vis)

        if only_number:
            return jsonify({'count': len(objects)})

        path_public = 'static/' + path_img_vis.split('static/')[-1]
        return jsonify({'objects': objects, 'count': len(objects),
                        'image_vis': path_public, 'mod_img': {'outsetx': outsetx, 'outsety': outsety, 'scale': scale}})
    except Exception as e:
        return jsonify(error='Unkown error', data=str(e), count=-666)


def start_network(config):
    global PREDICTOR_NETWORK
    try:
        PREDICTOR_NETWORK = PredictorNetwork(config)
    except Exception as e:
        # An error occurred loading the model; interrupt the whole server.
        tf.logging.error(e)
        _thread.interrupt_main()


@click.command(help='Start basic web application.')
@click.option('config_files', '--config', '-c', multiple=True, help='Config to use.')  # noqa
@click.option('--checkpoint', help='Checkpoint to use.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
@click.option('--host', default='0.0.0.0', help='Hostname to listen on. Set this to "0.0.0.0" to have the server available externally.')  # noqa
@click.option('--port', default=5000, help='Port to listen to.')
@click.option('--debug', is_flag=False, help='Set debug level logging.')
@click.option('--min-prob', default=0.5, type=float, help='Only get bounding boxes with probability larger than.')  # noqa
@click.option('--save-path', default="./luminoth/serve/static/output/", help='Put the location to save images predicted')  # noqa
def web(config_files, checkpoint, override_params, host, port, debug, min_prob, save_path):
    global SAVE_PATH_GLOBAL
    global SAVE_PATH_TMP
    SAVE_PATH_TMP = '/tmp/predict.jpg'

    if save_path:
        SAVE_PATH_GLOBAL = save_path
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    if checkpoint:
        config = get_checkpoint_config(checkpoint)
    elif config_files:
        config = get_config(config_files)
    else:
        raise ValueError(
            'Neither checkpoint not config specified, assuming `accurate`.'
        )

    if override_params:
        config = override_config_params(config, override_params)

    # Bounding boxes will be filtered by frontend (using slider), so we set a
    # low threshold.
    if config.model.type == 'fasterrcnn':
        config.model.rcnn.proposals.min_prob_threshold = min_prob
    elif config.model.type == 'ssd':
        config.model.proposals.min_prob_threshold = min_prob
    else:
        raise ValueError(
            "Model type '{}' not supported".format(config.model.type)
        )

    # Verfy folder path or create
    try:
        os.stat(SAVE_PATH_GLOBAL)
    except:
        os.mkdir(SAVE_PATH_GLOBAL)

    # Initialize model
    global NETWORK_START_THREAD
    NETWORK_START_THREAD = Thread(target=start_network, args=(config,))
    NETWORK_START_THREAD.start()

    if debug:
        app.config.from_object('config.DebugConfig')
    else:
        app.config.from_object('config.ProductionConfig')

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    web()

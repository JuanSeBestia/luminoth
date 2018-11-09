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

app = Flask(__name__)

def get_image_url():
    image = request.args.get("image",None)
    if not image:
        raise ValueError
    print("image",image)
    file_name = '/tmp/predict.jpg'
    urllib.request.urlretrieve(image, file_name)

    with tf.gfile.Open(file_name, 'rb') as f:
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
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/<model_name>/predict/', methods=['GET', 'POST'])
def predict(model_name):
    try:
        # TODO ADD more models
        if request.method == 'GET':
            print(request.args)
            #return jsonify(error='Use POST method to send image.', count=-1)
            total_predictions = request.args.get('total', None)
            min_prob = request.args.get('min_prob', None)
            only_number = request.args.get('only_number', False)
            id_task = request.args.get('id', False)
            
            try:
                image_array = get_image_url()
            except ValueError:
                return jsonify(error='Missing image', count=-2)
            except OSError:
                return jsonify(error='Incompatible file type', count=-3)
        else: #POST
            total_predictions = request.form.get('total', None)
            min_prob = request.form.get('min_prob', None)
            only_number = request.form.get('only_number', False)
            id_task = request.form.get('id', False)
            try:
                image_array = get_image()
            except ValueError:
                return jsonify(error='Missing image', count=-2)
            except OSError:
                return jsonify(error='Incompatible file type', count=-3)

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


        

        # Wait for the model to finish loading.
        NETWORK_START_THREAD.join()

        objects = PREDICTOR_NETWORK.predict_image(image_array)

        if min_prob:
            objects = [obj for obj in objects if obj['prob'] >= min_prob]

        if total_predictions:
            objects = objects[:total_predictions]

        # Save predicted image.
        if SAVE_PATH_GLOBAL:
            print("{}{}_Counted.jpg".format(SAVE_PATH_GLOBAL,id_task))
            vis_objects(np.array(image_array), objects).save("{}{}_Counted.jpg".format(SAVE_PATH_GLOBAL,id_task))

        if only_number:
            return jsonify({'count': len(objects)})
        
        return jsonify({'objects': objects, 'count': len(objects)})
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
@click.option('--save-path', default="./output/", help='Put the location to save images predicted')  # noqa

def web(config_files, checkpoint, override_params, host, port, debug, min_prob, save_path):
    global SAVE_PATH_GLOBAL
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
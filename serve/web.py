import click
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from threading import Thread
from PIL import Image
from six.moves import _thread

from luminoth.tools.checkpoint import get_checkpoint_config
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork

app = Flask(__name__)

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
            return jsonify(error='Use POST method to send image.', count=-1)

        total_predictions = request.form.get('total', None)
        min_prob = request.form.get('min_prob', None)
        only_number = request.form.get('only_number', False)
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

        try:
            image_array = get_image()
        except ValueError:
            return jsonify(error='Missing image', count=-2)
        except OSError:
            return jsonify(error='Incompatible file type', count=-3)

        # Wait for the model to finish loading.
        NETWORK_START_THREAD.join()

        objects = PREDICTOR_NETWORK.predict_image(image_array)

        if min_prob:
            objects = [obj for obj in objects if obj['prob'] >= min_prob]

        if total_predictions:
            objects = objects[:total_predictions]
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
def web(config_files, checkpoint, override_params, host, port, debug, min_prob):
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
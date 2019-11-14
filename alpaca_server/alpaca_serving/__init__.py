import multiprocessing
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
import os.path
import json

import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from alpaca_serving.helper import *
from alpaca_serving.httpproxy import HTTPProxy
from alpaca_serving.zmq_decor import multi_socket
from alpaca_model.pytorchAPI import SequenceTaggingModel

__all__ = ['__version__']
__version__ = '1.0.1'

# TODO
# Figure out a way to use this / enforce a schema - use of a function seems strange
# Communication_Message = namedtuple('Communication_Message', ["client_id", "user_id", "req_id", "msg_type", "msg_len", "msg"])
# Job_Message = namedtuple("Job_Message", ["job_id", "msg_type", "msg"])

class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    initiate = b'INITIATE'
    online_initiate = b'ONLINE_INITIATE'
    online_learning = b'ONLINE_LEARNING'
    active_learning = b'ACTIVE_LEARNING'
    predict = b'PREDICT'
    load = b'LOAD'
    error = b'ERROR'

    worker_status = b'WORKER_STATUS'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


# Ventilator
# Ventilator pushes data to workers with PUSH pattern.
class AlpacaServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'))
        self.args = args

        # ZeroMQ server configuration
        self.num_worker = args.num_worker  # number of Workers

        # restrict number of workers for temporaly
        self.num_concurrent_socket = max(16, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port

        # project configuration
        self.model_dir = args.model_dir  # alpaca_model per project
        self.models = {} # pass this model to every sink and worker!!!!
                        # in reality this is only being passed to every worker
        
        # learning initial configuration
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        self.status_args = {k: v for k, v in sorted(vars(args).items())}
        self.status_static = {
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }

        self.processes = []
        self.logger.info('Initialize the alpaca_model... could take a while...')
        self.server_event_obj = threading.Event()

    def __enter__(self):
        self.start()
        self.server_event_obj.wait()
        self.logger.info("Initial processes running")
        self.logger.info(str({'num_process': len(self.processes),
                              'processes' : self.processes,
                              'num_concurrent_socket': self.num_concurrent_socket}))
        self.logger.info("The working directory for the server: %s" % os.getcwd())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        self.server_event_obj.clear()
        self.join()

    # TODO
    # don't think this is being used properly
    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, client_sock):
        client_sock.connect('tcp://localhost:%d' % self.port)
        client_sock.send_multipart([b'0', b'0', b'0', ServerCmd.terminate])

    # TODO
    # don't think this is being used properly
    # what about closing down Sink and Worker processes?
    @staticmethod
    def shutdown(args):
        with zmq.Context() as ctx:
            ctx.setsockopt(zmq.LINGER, args.timeout)
            with ctx.socket(zmq.PUSH) as client_sock:
                try:
                    client_sock.connect('tcp://%s:%d' % (args.ip, args.port))
                    client_sock.send_multipart([b'0', b'0', b'0', ServerCmd.terminate])
                    print('shutdown signal sent to %d' % args.port)
                except zmq.error.Again:
                    raise TimeoutError(
                        'no response from the server (with "timeout"=%d ms), please check the following:'
                        'is the server still online? is the network broken? are "port" correct? ' % args.timeout)

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, client_sock, sink_sock, *worker_socks):
        # bind all sockets
        self.logger.info('bind all sockets')
        client_sock.bind('tcp://*:%d' % self.port)
        ventilator_sink_addr = auto_bind(sink_sock)
        backend_addr_list = [auto_bind(socket) for socket in worker_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(backend_addr_list))

        # start the sink process
        self.logger.info('start the sink')
        sink_proc = AlpacaSink(self.args, ventilator_sink_addr)
        self.processes.append(sink_proc)
        sink_proc.start()
        sink_pull_addr = sink_sock.recv().decode('ascii')

        # start the backend processes
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = AlpacaWorker(idx, self.args, self.models, backend_addr_list, sink_pull_addr, device_id)
            self.processes.append(process)
            process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = HTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_worker_socket = None
        server_status = ServerStatistic()

        # TODO
        # this needs to be fixed, the type of the object changes depending
        # on where the code is being run (import issues)
        for p in self.processes:
            if "AlpacaSink" in str(type(p)):
                p.sink_event_obj.wait()
            else:
                p.worker_event_obj.wait()

        self.server_event_obj.set()
        self.logger.info('all set, ready to serve request!')

        # receive message from client
        # make commands (1.recommend, 2.online learning(training) 3.kerasAPI learning ...)
        # project based file management
        while True:
            try:
                request = client_sock.recv_multipart()
                client_id, user_id, req_id, msg_type, msg_len, msg = request
                assert req_id.isdigit()
                assert msg_len.isdigit()
            except (ValueError, AssertionError):
                self.logger.error('received a wrongly-formatted request (expected 6 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg_type == ServerCmd.terminate:
                    break
                elif msg_type == ServerCmd.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client_id))
                    status_runtime = {'client': client_id.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'processes' : [type(process) for process in self.processes],
                                      'ventilator -> worker': backend_addr_list,
                                      'worker -> sink': sink_pull_addr,
                                      'ventilator <-> sink': ventilator_sink_addr,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}
                    
                    msg = jsonapi.dumps({**status_runtime,
                                         **self.status_args,
                                         **self.status_static})

                    # skip worker and send to sink
                    sink_sock.send_multipart([client_id, user_id, req_id, msg_type, msg])
                else:
                    self.logger.info('new %s request from user: %d on client: %s\treq id: %d\tsize: %d' % (msg_type, int(user_id), client_id, int(req_id), int(msg_len)))

                    # TODO
                    # Not sure we need this, this seems to involve letting small messages use sock[0] and larger messages
                    # should use other sockets, but:
                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backend shouldn't be selected either as it may be queued up already
                    rand_worker_socket = random.choice([b for b in worker_socks[1:] if b != rand_worker_socket])

                    # push a new job
                    job_id = client_id + b'#' + user_id + b'#' + req_id
                    try:
                        rand_worker_socket.send_multipart([job_id, msg_type, msg],zmq.NOBLOCK)  # fixed!
                        self.logger.info('%s job registered with id: %s and size: %d' % (msg_type, job_id, msg_len))
                    except zmq.error.Again:
                        # skip worker and sent this to straight to the sink
                        self.logger.info('zmq.error.Again: resource not available temporally, please send again!')
                        msg = jsonapi.dumps('zmq.error.Again: resource not available temporally, please send again!')
                        sink.send_multipart([client_id, user_id, req_id, ServerCmd.error, msg])

        for p in self.processes:
            p.close()
        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker),
                                                maxMemory=0.9, maxLoad=0.9)
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class AlpacaSink(Process):
    def __init__(self, args, ventilator_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'))
        self.ventilator_sink_addr = ventilator_sink_addr
        self.sink_event_obj = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.sink_event_obj.clear()
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, worker_receiver_sock, ventilator_sock, client_sock):
        logger = set_logger(colored('SINK', 'green'))

        logger.info("binding sink's three sockets")
        receiver_addr = auto_bind(worker_receiver_sock)
        ventilator_sock.connect(self.ventilator_sink_addr)
        client_sock.bind('tcp://*:%d' % self.port)
        logger.info("finishing binding sink's three sockets")

        logger.info("setting the ventilator and worker receiver sockets as Pollers")
        poller = zmq.Poller()
        poller.register(ventilator_sock, zmq.POLLIN)
        poller.register(worker_receiver_sock, zmq.POLLIN)
        logger.info("finsihed setting up Poller sockets")

        # TODO
        # send worker receiver address back to ventilator
        # Not sure why we do this, but we aren't handling this
        ventilator_sock.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger.info('sink ready')
        self.sink_event_obj.set()

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            # make sure worker sends this message
            if socks.get(worker_receiver_sock) == zmq.POLLIN:
                job_id, job_type, job_output = worker_receiver_sock.recv_multipart()

                client_id, user_id, req_id = job_id.split(b'#')
                if type(job_output) != bytes:
                    job_output = jsonapi.dumps(job_output)

                # make sure client recieves this data
                client_sock.send_multipart([client_id, user_id, req_id, job_output])
                logger.info('sending back job id: %s' % (job_id))

            # whole point of this is that certain requests don't require worker to process anything
            if socks.get(ventilator_sock) == zmq.POLLIN:
                client_id, user_id, req_id, msg_type, msg = ventilator_sock.recv_multipart()
                
                if msg_type == ServerCmd.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('sending server config to user: %d on client: %s' % (int(client_id), user_id))
                    client_sock.send_multipart([client_id, user_id, req_id, msg])
                if msg_type == ServerCmd.error:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('sending error to user: %d on client: %s' % (int(client_id), user_id))
                    client_sock.send_multipart([client_id, user_id, req_id, msg])

class AlpacaWorker(Process):
    def __init__(self, id, args, models, worker_address_list, sink_address, device_id):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'))
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.use_fp16 = args.fp16
        self.worker_event_obj = multiprocessing.Event()

        # we want server to have copy of model in case client quits
        # we can still save last updated model
        # in reality this should be done in a checkpoint way
        self.models = models
        self.model_ids = {}

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.worker_event_obj.clear()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    def status(self):
        """
        :rtype: dict[str, str]
        :return: a dictionary contains the status of this instance
        """
        return {
            "models" : str(self.models),
            "model_ids" : str(self.model_ids),
            "num_concurrent_socket" : str(self.num_concurrent_socket),
            "worker_id" : str(self.worker_id),
            "device_id" : str(self.device_id)
        }

    def _check_for_model(self, client_id, model_id):
        return client_id in self.models and model_id in self.models[user_id]

    def _model_not_found(self, sink_sock, job_id, msg_type, client_id, model_id, logger, error_output):
        logger.info('model can not be found, client: %s, model: %d' % (client_id, model_id))
        helper.send_test(sink_sock, job_id, msg_type, error_output)
        logger.info('%s job failed, job_id' % (msg_type, job_id))

    def _initiate_model(self, recv_sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger):
        passed_in_model_id = int(msg)
        logger.info('new %s job\tsocket: %d\tsize: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, 1, client_id, passed_in_model_id))

        model = SequenceTaggingModel()
        if user_id in models:
            client_model_path = "model_user_" + user_id + "_model_" + str(passed_in_model_id)
            if (os.path.isfile(os.path.join('.',client_model_path+'.pre')) and 
                os.path.isfile(os.path.join('.',client_model_path+'.pt'))):
                
                model.load(client_model_path)
                self.models[user_id][passed_in_model_id] = model
                self.models_ids[user_id][passed_in_model_id] = 1
                helper.send_test(sink_sock, job_id, msg_type, b'Model Loaded')
                logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, 1, client_id))
            else:
                self.models[user_id][passed_in_model_id] = model
                self.models_ids[user_id][passed_in_model_id] = 1
                helper.send_test(sink_sock, job_id, msg_type, b'Model Initiated')
                logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, 1, client_id))

        else:
            self.models[user_id] = {}
            self.model_ids[user_id] = {}
            self.models[user_id][passed_in_model_id] = model
            self.models_ids[user_id][passed_in_model_id] = 1
            helper.send_test(sink_sock, job_id, msg_type, b'Model Initiated')
            logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, 1, client_id))

    def _online_initiate(self, recv_sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger):
        model_id, sentences, labels = msg
        logger.info('new %s job\tsocket: %d\tsize: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, len(sentences), client_id, model_id))
        if _check_for_model(client_id, model_id):
            self.models[user_id][model_id].online_word_build(sentences, labels) # whole unlabeled training sentences / predefined_labels
            helper.send_test(sink_sock, job_id, msg_type, b'Online word build completed')
            logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, len(sentences), client_id))
        else:
            _model_not_found(sink_sock, job_id, msg_type, client_id, model_id, logger, b'Online word build failed')

    def _online_learning(self, recv_sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger):
        model_id, sentences, labels, epoch, batch = msg
        logger.info('new %s job\tsocket: %d\tsize: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, len(sentences), client_id, model_id))
        if _check_for_model(client_id, model_id):
            self.models[user_id][model_id].online_learning(sentences, labels, epoch, batch)
            client_model_path = "model_user_" + user_id + "_model_" + str(passed_in_model_id)
            self.models[user_id][model_id].save(client_model_path)
            helper.send_test(sink_sock, job_id, msg_type, b'Online learning completed')
            logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, len(sentences), client_id))
        else:
            _model_not_found(sink_sock, job_id, msg_type, client_id, model_id, logger, b'Online learning failed')
            

    def _predict(self, recv_sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger):
        model_id, sentences = msg
        logger.info('new %s job\tsocket: %d\tsize: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, len(sentences), client_id, model_id))
        if _check_for_model(client_id, model_id):
            analyzed_result = self.models[user_id][model_id].analyze(sentences)
            helper.send_test(sink_sock, job_id, msg_type, jsonapi.dumps(analyzed_result))
            logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, len(sentences), client_id))
        else:
            error_output = {
                'words': [],
                'entities': [],
                'tags': []
            }   
            _model_not_found(sink_sock, job_id, msg_type, client_id, model_id, logger, jsonapi.dumps(error_output))

    def _active_learning(self, recv_sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger):
        model_id, sentences, acquire = msg
        logger.info('new %s job\tsocket: %d\tsize: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, len(sentences), client_id, model_id))
        if _check_for_model(client_id, model_id):
            indices, scores = self.models[user_id][model_id].active_learning(sentences, acquire)
            json_indices = list(map(int, indices))
            json_scores = list(map(float, scores))
            active_data = {
                'indices': json_indices,
                'scores': json_scores,
            }
            helper.send_test(sink_sock, job_id, msg_type, jsonapi.dumps(active_data))
            logger.info('%s job done\tsize: %s\tclient: %s' % (msg_type, len(sentences), client_id))
        else:
            error_output = {
                'indices': [],
                'scores': []
            }   
            _model_not_found(sink_sock, job_id, msg_type, client_id, model_id, logger, jsonapi.dumps(error_output))

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink_sock, *ventilator_socks):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'))

        logger.info('use device %s' %
                    ('cpu' if self.device_id < 0 else 'gpu: %d' % self.device_id))

        logger.info("setting the ventilator receiver sockets as Pollers")
        poller = zmq.Poller()
        for sock, addr in zip(ventilator_socks, self.worker_address):
            sock.connect(addr)
            poller.register(sock, zmq.POLLIN)
        logger.info("finished setting up ventilator receiver sockets")

        logger.info("setting up connect to sink")
        sink_sock.connect(self.sink_address)
        logger.info("set up connection to sink")

        logger.info('ready and listening!')
        self.worker_event_obj.set()

        # logging here can be done better
        while not self.exit_flag.is_set():
            events = dict(poller.poll())
            for sock_idx, sock in enumerate(ventilator_socks):
                if sock in events:
                    job_id, msg_type, raw_msg = sock.recv_multipart()
                    client_id, user_id, req_id = job_id.split(b"#")
                    msg = jsonapi.loads(raw_msg)

                    if msg_type == ServerCmd.initiate:
                        _initiate_model(sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger)

                    elif msg_type == ServerCmd.online_initiate:
                        _online_initiate(sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger)

                    elif msg_type == ServerCmd.online_learning:
                        _online_learning(sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger)

                    elif msg_type == ServerCmd.predict:
                        _predict(sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger)

                    elif msg_type == ServerCmd.active_learning:
                        _active_learning(sock_idx, sink_sock, job_id, client_id, user_id, req_id, msg_type, msg, logger)

                    elif msg_type == ServerCmd.worker_status:
                        logger.info('new %s job\tsocket: %d\tclient: %s\tmodel: %d' % (msg_type, recv_sock_idx, client_id, model_id))
                        helper.send_test(sink_sock, job_id, msg_type, jsonapi.dumps(self.status()))
                        logger.info('%s job done\tclient: %s' % (msg_type, client_id))

class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client_id, user_id, req_id, msg_type, msg_len, msg = request
        self._hist_client[client_id] += 1
        if ServerCmd.is_valid(msg_type):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client_id] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client kerasAPI when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}
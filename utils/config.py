import yaml
from easydict import EasyDict
import os

from .logger import print_log

try:
    from petrel_client.client import Client
except ImportError:
    # raise ImportError('Please install petrel_client')
    logging.warning('Please install petrel_client''Please install petrel_client')

import io
import json

class CephManager:

    def __init__(self, s2_conf_path='~/petreloss.conf'):
        self.conf_path = s2_conf_path
        self._client = Client(conf_path=s2_conf_path)

    def readlines(self, url):

        response = self._client.get(url, enable_stream=True, no_cache=True)

        lines = []
        for line in response.iter_lines():
            lines.append(line.decode('utf-8'))
        return lines

    def load_data(self, path, ceph_read=False):
        if ceph_read:
            return self.readlines(path)
        else:
            return self._client.get(path)

    def get(self, file_path):
        return self._client.get(file_path)


    def load_json(self, json_url):
        return json.loads(self.load_data(json_url, ceph_read=False))


    def write(self, save_dir, obj):
        self._client.put(save_dir, obj)

    def put_text(self,
                 obj: str,
                 filepath,
                 encoding: str = 'utf-8') -> None:
        self.write(filepath, bytes(obj, encoding=encoding))

    def exists(self, url):
        return self._client.contains(url)
    
    def remove(self, url):
        return self._client.delete(url)
    
    def isdir(self, url):
        return self._client.isdir(url)

    def isfile(self, url):
        return self.exists(url) and not self.isdir(url)

    def listdir(self, url):
        return self._client.list(url)

    def copy(self, src_path, dst_path, overwrite):
        if not overwrite and self.exists(dst_path):
            pass
        object = self._client.get(src_path)
        self._client.put(dst_path, object)
        return dst_path
    
def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    client = CephManager()
    if ":" in cfg_file:
        try:
            new_config = yaml.load(client.get(cfg_file), Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(client.get(cfg_file))
    else:
        with open(cfg_file, 'r') as f:  
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)
            
    merge_new_config(config=config, new_config=new_config)        
    return config

def get_config(args, logger=None):
    if args.resume:
        client = CephManager()
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if ":" in cfg_path:
            if not client.exists(cfg_path):
                print_log("Failed to resume", logger = logger)
                raise FileNotFoundError()
        else:
            if not os.path.exists(cfg_path):
                print_log("Failed to resume", logger = logger)
                raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    if ":" in config_path:
        os.system('aws s3 cp %s %s/config.yaml' % (args.config, args.experiment_path))
    else:
        os.system("cp %s %s" % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}', logger = logger )    
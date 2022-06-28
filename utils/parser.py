import os
import argparse
from pathlib import Path
try:
    from petrel_client.client import Client
    import json, io, torch
except ImportError:
    # raise ImportError('Please install petrel_client')
    logging.warning('Please install petrel_client''Please install petrel_client')
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

    def load_model(self, model_path, map_location):
        file_bytes = self._client.get(model_path)
        buffer = io.BytesIO(file_bytes)
        return torch.load(buffer, map_location=map_location)

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
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')  
    parser.add_argument(
        '--save_dir',
        default='./pointr_experiments',
        type=str
    )      
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join(args.save_dir, Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join(args.save_dir, Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    client = CephManager()
    if not client.exists(args.experiment_path):

        print('Create experiment path successfully at %s' % args.experiment_path)
    mapping_path = {'s3://NLP/jsy': "/mnt/lustre/jiangshuyang"}
    args.tfboard_path = str.replace(args.tfboard_path, 's3://NLP/jsy', mapping_path['s3://NLP/jsy'])
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_



from pathlib import Path
import natsort

from tqdm import tqdm
import json
import decord
import imageio

from huggingface_hub import hf_hub_download

S3FD_WEIGHT_REPO = "lithiumice/syncnet"

class S3FD():

    def __init__(self, device='cuda', **kwargs):

        tstamp = time.time()
        self.device = device

        self.net = S3FDNet(device=self.device).to(self.device)
        
        PATH_WEIGHT = hf_hub_download(repo_id=S3FD_WEIGHT_REPO, filename="sfd_face.pth")
        assert Path(PATH_WEIGHT).exists(), f'{PATH_WEIGHT=} not exists.'
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        print(f"[S3FD] loaded weight from {PATH_WEIGHT}, to device {self.device}")
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
        
        self.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        if (s_scale := (w*h)/(1080*1920))>1:
            # print(f'!!! large img resize factor=0.5')
            scales = [ii/s_scale for ii in scales]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= self.img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes
    
    def infer_img(self, im_path, facedet_scale=0.25):
        # import ipdb;ipdb.set_trace()
        image = imageio.imread(im_path)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.detect_faces(image_np,conf_th=0.9,scales=[facedet_scale])
        return bboxes

    def infer_img_dir(self, img_list_dir, facedet_scale=0.25):
        # import ipdb;ipdb.set_trace()
        
        img_dir = Path(img_list_dir)
        flist = [str(_) for _ in img_dir.glob("*.jpg")]
        flist = natsort.natsorted(flist)
        
        dets = []
        for fidx, fname in (jbar:=tqdm(enumerate(flist), desc='infer S3FD', leave=True)):
            bboxes = self.infer_img(fname)
            # start_time = time.time()
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({
                    'frame': fidx,
                    'bbox': (bbox[:-1]).tolist(),
                    'conf': bbox[-1]
                })

            # elapsed_time = time.time() - start_time
            
        return dets

    def infer_video(self, video_path, facedet_scale=0.25):
        
        video_path = str(video_path)
        
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))

        return self.infer_vr(vr, facedet_scale=facedet_scale)
        
    def infer_vr(self, vr, facedet_scale=0.25):

        dets = []
        
        for fidx in (jbar:=tqdm(range(len(vr)), desc='infer S3FD', leave=True)):
            image = vr[fidx].asnumpy()
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)            
            
            bboxes = self.detect_faces(image,conf_th=0.9,scales=[facedet_scale])

            dets.append([])
            for bbox in bboxes:
                dets[-1].append({
                    'frame': fidx,
                    'bbox': (bbox[:-1]).tolist(),
                    'conf': bbox[-1]
                })
                
        return dets
                
    def infer_video_save_json(self, video_path):
        video_path = str(video_path)
        
        dets = self.infer_video(video_path)
        json_path = video_path.replace('.mp4','.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(dets))
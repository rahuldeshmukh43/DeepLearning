# model file contains routines for getting input, passing through network, computing loss, back prop
# optimizer steps, training loop, printing stuff, logging, saving weights
import os
import numpy as np
import torch
from torch import optim
from torch.nn import DataParallel
from torch.autograd import Variable
import torchvision.utils as vutils

from src.NCN.network import NCNet
from src.NCN.NCN_config import get_cfg
from src.NCN.criterion import NCNet_Loss
import utils


class NCNetModel():

    def name(self):
        return 'NC Net Model'

    def __init__(self, args, logger):
        # SAVE ARGS
        self.args = args
        self.logger = logger
        self.config = get_cfg()
        self.flag_multi_gpu = args.multi_gpu

        # SET DEVICE
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_ids = [id for id in range(torch.cuda.device_count())]

        # INIT MODEL, OPTIMIZER, SCHEDULER
        if not self.flag_multi_gpu:
            self.model = NCNet(args, self.config, self.device)
        else:
            self.model = DataParallel(NCNet(args, self.config, self.device),
                                      device_ids=self.device_ids)
        # create a list of trainable parameters:
        params_to_update = self.model.parameters()
        if args.fix_backbone:
            params_to_update = []
            for param in self.model.parameters():
                if param.requires_grad:
                    params_to_update.append(param)

        self.optimizer = optim.Adam(params_to_update, lr=self.args.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=args.lrate_decay_steps,
        #                                                  gamma=args.lrate_decay_factor)

        # RELOAD FROM CHECKPOINT
        self.start_step = self.load_from_checkpoint()

        # DEFINE LOSS FUNCTION
        self.criterion = NCNet_Loss().to(self.device)

    def set_input(self, data):
        "load input from data to device, load ancilliary data for loss computation, plotting, etc"
        self.imgA = Variable(data['img1'].to(self.device, dtype=torch.float))
        self.imgB = Variable(data['img2'].to(self.device, dtype=torch.float))
        self.gt_label = data['label'].to(self.device, dtype=torch.float)
        # self.class_idxA = data['class_idx1']
        # self.class_idxB = data['class_idx2']
        # self.imgA_name = data['img1_name']
        # self.imgB_name = data['img2_name']
        self.imgA_ori = data['img1_ori']
        self.imgB_ori = data['img2_ori']
        self.imgA_annotation = data['img1_annotation']
        self.imgB_annotation = data['img2_annotation']
        self.common_kp_idx = data['common_kp_idx']
        self.imgA_size = data['img1_size']
        self.imgB_size = data['img2_size']
        self.imsize = self.imgA.size()[2:]  # HW

    def forward(self):
        self.out, self.matches, self.score_A, self.score_B, self.mean_score_A, self.mean_score_B = self.model.forward(
            self.imgA, self.imgB)  # self.out is the 4D correlation
        self.corr_size = self.out.size()[-1]

    def compute_loss(self):
        # use self.out, self.label and self.criterion to compute the loss
        self.loss = self.criterion(self.out, self.gt_label,
                                   self.mean_score_A, self.mean_score_B)

    def backward_net(self):
        # do backwards on the loss
        torch.autograd.set_detect_anomaly(True)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.compute_loss()
        self.backward_net()
        self.optimizer.step()
        # self.scheduler.step()

    def test(self):
        """run model during testing"""
        # TODO: write testing routine
        self.model.eval()
        with torch.no_grad():
            # matches, matching_score = self.model.test(self.imgA, self.imgB)
            self.forward()
        return  # matches, matching_score

    def threshold_matches(self, matches: torch.tensor, matching_score: torch.tensor, threshold: float) -> list:
        """ threshold the scores to get best matches
        :param matches  [B, H1, W1, 2]
        :param matching_score [B,H1,W1,H2,W2]
        :param threshold float
        :return filtered_matches nested list with first dimension as batch B, 
                for each batch its a list of (i,j,k,l,score) tupele
        """
        B, H1, W1, H2, W2 = matching_score.shape
        thresholded_matching_scores = torch.where(matching_score > threshold,
                                                  matching_score,
                                                  torch.tensor([0.]).to(self.device))
        filtered_matches = []
        for ib in range(B):
            this_filtered_matches = []
            for i in range(H1):
                for j in range(W1):
                    k, l = matches[ib, i, j, :]
                    s = thresholded_matching_scores[ib, i, j, k, l]
                    if s > 0.:
                        this_filtered_matches.append((i, j, k, l, s))
            filtered_matches.append(this_filtered_matches)

        # TODO: scale back match locations to original image sizes
        return filtered_matches

    def compute_eval_score(self, matches, threshold=1):
        """
        Computes the PCK (Percentage correct keypoints) score using annotations
        :param matches [B, H1, W1, 2]
        :param threshold int, distance (in pixels) threshold for assesing correct keypoint
        """
        pck_score = 0.0
        B = self.imgA.shape[0]

        matches = matches.cpu().numpy()

        # compute matches corresponding to annotations in imgA
        # i)scale annotations to current image size
        kps_A = [self.imgA_annotation[idx]['kps'][self.common_kp_idx[idx]] for idx in range(B)]
        kps_B = [self.imgB_annotation[idx]['kps'][self.common_kp_idx[idx]] for idx in range(B)]
        num_kps = [kps_A[idx].shape[0] for idx in range(B)]
        predicted_matches_B = []
        for idx in range(B):
            # imgA
            kps_A[idx][:, 0] *= 1.0 * (self.corr_size / self.imgA_size[idx][1])  # x (W)
            kps_A[idx][:, 1] *= 1.0 * (self.corr_size / self.imgA_size[idx][0])  # y (H)
            # imgB
            kps_B[idx][:, 0] *= 1.0 * (self.corr_size / self.imgB_size[idx][1])  # x (W)
            kps_B[idx][:, 1] *= 1.0 * (self.corr_size / self.imgB_size[idx][0])  # y (H)
            # convert to int
            kps_A[idx] = np.floor(kps_A[idx]).astype(np.int32)
            # get matching location using matches
            this_predicted_matches_B = np.array([matches[idx][kp[1]][kp[0]][:] for kp in kps_A[idx]])
            predicted_matches_B.append(this_predicted_matches_B)

        # compare predicted locations with true locations
        for idx in range(B):
            kp_B = kps_B[idx]
            predicted_kp_B = predicted_matches_B[idx]
            dist = np.linalg.norm((kp_B - predicted_kp_B), axis=1)
            pck_score += np.sum(dist < threshold)

        pck_score /= np.sum(num_kps)
        return pck_score

    def write_train_summary(self, writer, n_iter, epoch, i_minibatch):
        self.logger.info("Training: %s | Step: %d [Epoch:%d batch:%d], Loss: %2.5f" % (self.args.exp_name,
                                                                                       n_iter, epoch, i_minibatch,
                                                                                       self.loss.item()))

        # WRITE SCALAR
        if n_iter % self.args.log_scalar_interval == 0:
            writer.add_scalar('Loss', self.loss.item(), n_iter)

            # WRITE IMAGE
        if n_iter % self.args.log_img_interval == 0:
            if torch.any(self.gt_label == 1):
                # get index for gt_label==1
                idx = (self.gt_label == 1).nonzero(as_tuple=True)[0][0]
            else:
                return
            # this visualization shows the semantic keypoint transfer for annotaions available for images
            # note: when label is not 1 then the matches will not make any sense as they are of different object clas

            # convert images to numpy
            im1 = self.imgA_ori[idx].cpu().numpy()  # CHW
            im2 = self.imgB_ori[idx].cpu().numpy()

            # convert CHW images to HWC as cv2 requires that
            im1 = im1.transpose(1, 2, 0)
            im2 = im2.transpose(1, 2, 0)

            # compute matches corresponding to annotations in imgA
            # i)scale annotations to current image size
            kps_A = self.imgA_annotation[idx]['kps']
            notnan_idxA = self.imgA_annotation[idx]['notnan_idx']
            kps_A = kps_A[notnan_idxA, :]
            kps_A_scaled = np.copy(kps_A)
            num_kps = self.imgA_annotation[idx]['num_kps']
            # scale kps_A to imsize for plotting
            kps_A[:, 0] *= 1.0 * (self.imsize[1] / self.imgA_size[idx][1])  # x (W)
            kps_A[:, 1] *= 1.0 * (self.imsize[0] / self.imgA_size[idx][0])  # y (H)

            # scale kps_A to corr size --- to get corresponding matches
            kps_A_scaled[:, 0] *= 1.0 * (self.corr_size / self.imgA_size[idx][1])  # x (W)
            kps_A_scaled[:, 1] *= 1.0 * (self.corr_size / self.imgA_size[idx][0])  # y (H)

            # ii) convert to int
            kps_A_scaled = np.floor(kps_A_scaled).astype(np.int)

            # iii) get matching location using self.matches
            matches, _matching_score = self.get_matches_and_matching_score()  # matches stores (y,x)
            matches = matches.cpu().numpy()
            predicted_matches_B_scaled = np.array(
                [matches[idx][kp[1]][kp[0]][::-1] for kp in kps_A_scaled])  # (x,y) tuples list
            predicted_matches_B = np.copy(predicted_matches_B_scaled)

            # scale back predicted matches to imsize
            predicted_matches_B[:, 0] *= 1.0 * (self.imsize[1] / self.corr_size)  # x (W)
            predicted_matches_B[:, 1] *= 1.0 * (self.imsize[0] / self.corr_size)  # y (H)

            # draw matches and true matches
            img = utils.draw_matches(im1, im2, kps_A, predicted_matches_B)  # returns #CHW

            # save image to tensorboard
            img = torch.from_numpy(img).float().unsqueeze(0)
            x = vutils.make_grid(img)  # , normalize=True)
            writer.add_image('Image', x, n_iter)

        return

    def write_eval_summary(self, writer, n_iter, epoch):
        self.logger.info("Validation: %s | [Epoch: %d batch: %d] Loss: %2.5f" % (self.args.exp_name,
                                                                                 epoch, n_iter,
                                                                                 self.loss.item()))

        # WRITE SCALAR
        if n_iter % self.args.log_scalar_interval_eval == 0:
            writer.add_scalar('Validation Loss', self.loss.item(), n_iter)

        return

    def write_epoch_summary(self, writer, epoch, avg_loss, avg_vloss, avg_eval_score):
        self.logger.info("Model Summary: | [Epoch: %d] Training Loss: %2.5f "
                         "Validation Loss: %2.5f "
                         "Validation Score: %2.5f" % (epoch, avg_loss, avg_vloss, avg_eval_score))

        writer.add_scalar('Avg Loss', avg_loss, epoch)
        writer.add_scalar('Avg Validation Loss', avg_vloss, epoch)
        writer.add_scalar('Avg Eval Score', avg_eval_score, epoch)

        return

    def load_model(self, filename):
        to_load = torch.load(filename)
        self.model.load_state_dict(to_load['state_dict'])
        # if 'optimizer' in to_load.keys():
        #     self.optimizer.load_state_dict(to_load['optimizer'])
        # if 'scheduler' in to_load.keys():
        #     self.scheduler.load_state_dict(to_load['scheduler'])
        return to_load['step']

    def load_from_checkpoint(self):
        """
        load model from existing checkpoints and return the current step
        :return: the current starting step
        """

        # load from the specified ckpt path
        if self.args.ckpt_path != "":
            self.logger.info("Reloading from {}".format(self.args.ckpt_path))
            if os.path.isfile(self.args.ckpt_path):
                step = self.load_model(self.args.ckpt_path)
            else:
                raise Exception('no checkpoint found in the following path:{}'.format(self.args.ckpt_path))

        else:
            ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
            os.makedirs(ckpt_folder, exist_ok=True)
            # load from the most recent ckpt from all existing ckpts
            ckpts = [os.path.join(ckpt_folder, f) for f in sorted(os.listdir(ckpt_folder)) if f.endswith('.pth')]
            if len(ckpts) > 0:
                fpath = ckpts[-1]
                step = self.load_model(fpath)
                self.logger.info('Reloading from {}, starting at step={}'.format(fpath, step))
            else:
                self.logger.info('No ckpts found, training from scratch...')
                step = 0

        return step

    def save_model(self, step):
        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        save_path = os.path.join(ckpt_folder, "{:06d}.pth".format(step))
        self.logger.info('saving ckpts {}...'.format(save_path))
        torch.save({'step': step,
                    'state_dict': self.model.state_dict(),
                    # 'optimizer':  self.optimizer.state_dict(),
                    # 'scheduler': self.scheduler.state_dict(),
                    },
                   save_path)

    def get_loss(self):
        return self.loss.item()

    def get_matches_and_matching_score(self):
        return self.matches, self.score_B

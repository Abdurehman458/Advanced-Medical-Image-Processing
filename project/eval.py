import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete

from train import parse_args,DS
base = "./dataset"
results = './test_result/'
unet_results = './unet_results/'
def visualize():
    for img_name in sorted(os.listdir(base+"/test")):
        label_pth = os.path.join(base+"/test/")
        label = np.load(label_pth+img_name+'/seg.npy')
        label = np.clip(label,0,1)
        pred_label = Image.open(results+img_name+'.png')
        unet_label = Image.open(unet_results+img_name+'.png')
        plt.subplot(1,3,1),plt.imshow(label),plt.title('label')
        plt.subplot(1,3,2),plt.imshow(pred_label),plt.title('Fast-SCNN')
        plt.subplot(1,3,3),plt.imshow(unet_label),plt.title('U-NET')
        plt.show()


def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = 'test_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        # val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
        #                                        transform=input_transform)
        val_dataset = DS(base,datatype="test")
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        # create network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, root=args.save_folder).to(args.device)
        print('Finished loading model!')

        # self.metric = SegmentationMetric(val_dataset.num_class)
        self.metric = SegmentationMetric(2)

    def eval(self):
        self.model.eval()
        dice_scores = 0
        for i, (image,name, label) in enumerate(self.val_loader):
            image = image.to(self.args.device,dtype=torch.float32)

            outputs = self.model(image)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU = self.metric.get()
            print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (i + 1, pixAcc * 100, mIoU * 100))

            predict = pred.squeeze(0)
            mask = get_color_pallete(predict, self.args.dataset)
            mask.save(os.path.join(self.outdir, '{}.png'.format(name[0])))
            dice_score = compute_dice_coefficient(label.astype('uint8'),predict)
            dice_scores+=dice_score
        print("Avg Dice Score:", 100*dice_scores/84)

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()
    visualize()

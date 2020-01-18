import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import cv2 as cv


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                use_dense_crf=False):
    net.eval()

    ds = BasicDataset('', '', scale=scale_factor)
    img = torch.from_numpy(ds.preprocess(full_img))
    print('28 img', img.shape)
    img = img.unsqueeze(0)
    print('30 img', img.shape)
    img = img.to(device=device, dtype=torch.float32)

    # traced_script_module = torch.jit.trace(net, img)
    # 保存模型
    # traced_script_module.save("torch_script_eval.pth")

    with torch.no_grad():
        print('38 input into net', img.shape)
        output = net(img)
        print('40 output shape', output.shape)

        if net.module.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        print('48 probs shape', probs.shape)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(np.array(full_img).shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    full_mask[0][full_mask[0] > out_threshold] = 1.0
    full_mask[0][full_mask[0] <= out_threshold] = 0.0
    full_mask[1][full_mask[1] > out_threshold] = 1.0
    full_mask[1][full_mask[1] <= out_threshold] = 0.0
    full_mask[2][full_mask[2] > out_threshold] = 1.0
    full_mask[2][full_mask[2] <= out_threshold] = 0.0
    full_mask[3][full_mask[3] > out_threshold] = 1.0
    full_mask[3][full_mask[3] <= out_threshold] = 0.0
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    n_mask = np.array(mask * 20)
    for channel in range(0, 4):
        cv.imshow('prediction', n_mask[channel])
        cv.waitKey()
        cv.destroyAllWindows()
    # n_mask = n_mask.sum(axis=0)
    print('116 n_mask shape', n_mask.shape)

    # cv.imshow('prediction', n_mask)
    # cv.waitKey()

    return Image.fromarray(n_mask.astype(np.uint8))
    # print(mask.shape)


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = torch.nn.DataParallel(net)
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    # device = torch.device('cpu')
    # pretained_dict = torch.load(args.model)
    # net.load_state_dict(pretained_dict)  # CPU-version

    logging.info("Model loaded !")
    # torch.save(net.module, 'save_as.pth')

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=False,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
